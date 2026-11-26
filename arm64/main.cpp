// Native ARM64 inference path for the .asmbin format used by llm.asm.
// The model layout, Q8_0 math, tokenizer, and Qwen3 forward pass follow the
// original assembly implementation; NEON replaces the AVX2 dot-product core.
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kVocab = 151936;
constexpr uint32_t kImStart = 151644, kImEnd = 151645;
constexpr uint32_t kThinkStart = 151667, kThinkEnd = 151668;
constexpr uint32_t kUser = 872, kAssistant = 77091, kNl = 198, kNl2 = 271;
constexpr uint32_t kEmptyRank = 0xffffffffu;
constexpr uint64_t kFib = 0x9e3779b97f4a7c15ull;
constexpr size_t kHtSize = 1u << 19;

uint32_t u32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24; }
uint64_t u64(const uint8_t* p) { return uint64_t(u32(p)) | uint64_t(u32(p+4))<<32; }
size_t align64(size_t x) { return (x + 63) & ~size_t(63); }
float fp16(const uint8_t* p) {
    const uint16_t h=uint16_t(p[0])|uint16_t(p[1])<<8;
    const uint32_t sign=uint32_t(h&0x8000u)<<16;
    uint32_t exponent=(h>>10)&0x1fu;
    uint32_t mantissa=h&0x03ffu;
    uint32_t bits;
    if(exponent==0){
        if(mantissa==0) bits=sign;
        else {
            int e=-14;
            while((mantissa&0x0400u)==0){mantissa<<=1;--e;}
            mantissa&=0x03ffu;
            bits=sign|(uint32_t(e+127)<<23)|(mantissa<<13);
        }
    } else if(exponent==0x1fu) {
        bits=sign|0x7f800000u|(mantissa<<13);
    } else {
        bits=sign|((exponent+112u)<<23)|(mantissa<<13);
    }
    return std::bit_cast<float>(bits);
}

struct Matrix { const uint8_t* data{}; uint32_t rows{}, cols{}; size_t row_stride{}; };

struct Model {
    std::vector<uint8_t> file;
    const uint8_t *b2t{}, *tok_offsets{}, *tok_bytes{}, *ht{}, *embed{}, *layers{}, *final_norm{};
    uint32_t dim{}, hidden{}, n_layers{}, n_heads{}, n_kv{}, head_dim{}, vocab{}, max_seq{}, version{}, scale_bytes{};
    float eps{}, rope_theta{};
    size_t layer_stride{};
    uint32_t eos1{}, eos2{};
    explicit Model(const char* path) {
        std::ifstream f(path, std::ios::binary|std::ios::ate);
        if (!f) throw std::runtime_error(std::string("cannot open model: ")+path);
        auto sz=f.tellg(); if (sz<128) throw std::runtime_error("model file is too small");
        file.resize(size_t(sz)); f.seekg(0); f.read(reinterpret_cast<char*>(file.data()), sz);
        const uint8_t* h=file.data();
        if (std::memcmp(h,"ASML",4)!=0) throw std::runtime_error("not an ASML model");
        version=u32(h+4); if (version!=1 && version!=2) throw std::runtime_error("unsupported ASML version");
        scale_bytes=version==2?2:4;
        dim=u32(h+8); hidden=u32(h+12); n_layers=u32(h+16); n_heads=u32(h+20); n_kv=u32(h+24); head_dim=u32(h+28); vocab=u32(h+32); max_seq=u32(h+36);
        eps=std::bit_cast<float>(u32(h+40)); rope_theta=std::bit_cast<float>(u32(h+44)); eos1=u32(h+48); eos2=u32(h+52); layer_stride=u64(h+88);
        auto at=[&](uint64_t off)->const uint8_t* { if (off>=file.size()) throw std::runtime_error("invalid ASML offset"); return file.data()+off; };
        const uint8_t* tb=at(u64(h+64));
        b2t=tb+u64(tb); tok_offsets=tb+u64(tb+8); tok_bytes=tb+u64(tb+16); ht=tb+u64(tb+24);
        embed=at(u64(h+72)); layers=at(u64(h+80)); final_norm=at(u64(h+96));
    }
    size_t rs(uint32_t k) const { return size_t(k)+(size_t(k)/32)*scale_bytes; }
    const uint8_t* aligned(const uint8_t* p) const {
        size_t off=size_t(p-file.data()); return file.data()+align64(off);
    }
    Matrix matrix(const uint8_t*& p,uint32_t rows,uint32_t cols) const {
        p=aligned(p);
        Matrix m{p,rows,cols,rs(cols)}; p+=size_t(rows)*m.row_stride;
        p=aligned(p); return m;
    }
    std::array<Matrix,7> layer_mats(uint32_t l, std::array<const float*,4>& norms) const {
        const uint8_t* p=layers+size_t(l)*layer_stride;
        norms[0]=reinterpret_cast<const float*>(p); p+=dim*4;
        norms[1]=reinterpret_cast<const float*>(p); p+=head_dim*4;
        norms[2]=reinterpret_cast<const float*>(p); p+=head_dim*4;
        norms[3]=reinterpret_cast<const float*>(p); p+=dim*4;
        p=aligned(p);
        return {matrix(p,n_heads*head_dim,dim), matrix(p,n_kv*head_dim,dim), matrix(p,n_kv*head_dim,dim),
                matrix(p,dim,n_heads*head_dim), matrix(p,hidden,dim), matrix(p,hidden,dim), matrix(p,dim,hidden)};
    }
    uint32_t merge(uint32_t a,uint32_t b,uint32_t& rank) const {
        uint64_t key=(uint64_t(a)<<32)|b, slot=(key*kFib)>>(64-19);
        for (;;) { const uint8_t* e=ht+slot*16; uint32_t r=u32(e+12); if(r==kEmptyRank){rank=r;return 0;} if(u64(e)==key){rank=r;return u32(e+8);} slot=(slot+1)&(kHtSize-1); }
    }
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> t; t.reserve(text.size());
        for(uint8_t c:text) t.push_back(u32(b2t+c*4));
        while(t.size()>1) {
            uint32_t best=kEmptyRank, best_tok=0; size_t best_i=t.size();
            for(size_t i=0;i+1<t.size();++i){uint32_t r;uint32_t m=merge(t[i],t[i+1],r);if(r<best){best=r;best_tok=m;best_i=i;}}
            if(best_i==t.size()) break; t[best_i]=best_tok;t.erase(t.begin()+best_i+1);
        }
        return t;
    }
    std::string token(uint32_t id) const {
        if(id>=vocab)return {}; uint32_t a=u32(tok_offsets+4*id),b=u32(tok_offsets+4*(id+1)); return std::string(reinterpret_cast<const char*>(tok_bytes+a),b-a);
    }
};

struct QVec { std::vector<int8_t> q; std::vector<float> s; };
QVec quantize(const float* x,uint32_t n) {
    QVec q; q.q.resize(n); q.s.resize(n/32);
    for(uint32_t b=0;b<n/32;++b){float a=0;for(uint32_t j=0;j<32;++j)a=std::max(a,std::abs(x[b*32+j]));float s=a/127.f;q.s[b]=s;float inv=s?1.f/s:0.f;for(uint32_t j=0;j<32;++j)q.q[b*32+j]=int8_t(std::clamp(std::nearbyint(x[b*32+j]*inv),-127.f,127.f));}
    return q;
}
int32_t dot32(const int8_t* a,const int8_t* b) {
    int8x16_t a0=vld1q_s8(a),a1=vld1q_s8(a+16),b0=vld1q_s8(b),b1=vld1q_s8(b+16);
    int16x8_t p0=vmull_s8(vget_low_s8(a0),vget_low_s8(b0));
    int16x8_t p1=vmull_s8(vget_high_s8(a0),vget_high_s8(b0));
    int16x8_t p2=vmull_s8(vget_low_s8(a1),vget_low_s8(b1));
    int16x8_t p3=vmull_s8(vget_high_s8(a1),vget_high_s8(b1));
    int32x4_t s=vaddq_s32(vaddq_s32(vpaddlq_s16(p0),vpaddlq_s16(p1)),vaddq_s32(vpaddlq_s16(p2),vpaddlq_s16(p3)));
    return vaddvq_s32(s);
}
void matvec(const Model& m,const Matrix& w,const float* x,float* y) {
    QVec q=quantize(x,w.cols); const uint32_t blocks=w.cols/32;
    for(uint32_t r=0;r<w.rows;++r){const uint8_t* row=w.data+size_t(r)*w.row_stride;float sum=0;
        for(uint32_t b=0;b<blocks;++b){float ws=m.scale_bytes==2?fp16(row+w.cols+2*b):std::bit_cast<float>(u32(row+w.cols+4*b));sum+=float(dot32(reinterpret_cast<const int8_t*>(row+32*b),q.q.data()+32*b))*ws*q.s[b];} y[r]=sum;}
}
void dequant(const Model& m,const uint8_t* row,float* out,uint32_t n){for(uint32_t b=0;b<n/32;++b){float s=m.scale_bytes==2?fp16(row+n+2*b):std::bit_cast<float>(u32(row+n+4*b));for(int j=0;j<32;++j)out[b*32+j]=int8_t(row[b*32+j])*s;}}
void rmsnorm(float* out,const float* x,const float* w,uint32_t n,float eps){double ss=0;for(uint32_t i=0;i<n;++i)ss+=double(x[i])*x[i];float k=1.f/std::sqrt(float(ss/n)+eps);for(uint32_t i=0;i<n;++i)out[i]=x[i]*k*w[i];}
void add(float* x,const float* y,uint32_t n){for(uint32_t i=0;i<n;++i)x[i]+=y[i];}
float silu(float x){if(x>=0.f)return x/(1.f+std::exp(-x));float e=std::exp(x);return x*e/(1.f+e);}

struct Engine {
    const Model& m; uint32_t qdim, kvdim, maxseq; std::vector<float> x,xb,q,k,v,att,gate,up,logits,cosv,sinv;
    std::vector<float> kc,vc,scores; std::vector<double> invfreq;
    explicit Engine(const Model& model,uint32_t ctx):m(model),qdim(m.n_heads*m.head_dim),kvdim(m.n_kv*m.head_dim),maxseq(std::min(ctx,m.max_seq)),x(m.dim),xb(m.dim),q(qdim),k(kvdim),v(kvdim),att(qdim),gate(m.hidden),up(m.hidden),logits(m.vocab),cosv(m.head_dim/2),sinv(m.head_dim/2),kc(size_t(m.n_layers)*m.n_kv*maxseq*m.head_dim),vc(kc.size()),scores(maxseq),invfreq(m.head_dim/2){for(uint32_t i=0;i<m.head_dim/2;++i)invfreq[i]=std::pow(double(m.rope_theta),-2.0*i/m.head_dim);}
    float* cache(std::vector<float>& c,uint32_t l,uint32_t h,uint32_t pos){return c.data()+(((size_t(l)*m.n_kv+h)*maxseq+pos)*m.head_dim);}
    void forward(uint32_t tok,uint32_t pos,bool calc){
        if(pos>=maxseq)throw std::runtime_error("context full; clear the conversation");
        size_t emb_stride=m.rs(m.dim);dequant(m,m.embed+size_t(tok)*emb_stride,x.data(),m.dim);
        for(uint32_t i=0;i<m.head_dim/2;++i){double a=double(pos)*invfreq[i];cosv[i]=float(std::cos(a));sinv[i]=float(std::sin(a));}
        float scale=1.f/std::sqrt(float(m.head_dim));
        for(uint32_t l=0;l<m.n_layers;++l){std::array<const float*,4> n;auto w=m.layer_mats(l,n);
            rmsnorm(xb.data(),x.data(),n[0],m.dim,m.eps);matvec(m,w[0],xb.data(),q.data());matvec(m,w[1],xb.data(),k.data());matvec(m,w[2],xb.data(),v.data());
            for(uint32_t h=0;h<m.n_heads;++h)rmsnorm(q.data()+size_t(h)*m.head_dim,q.data()+size_t(h)*m.head_dim,n[1],m.head_dim,m.eps);
            for(uint32_t h=0;h<m.n_kv;++h)rmsnorm(k.data()+size_t(h)*m.head_dim,k.data()+size_t(h)*m.head_dim,n[2],m.head_dim,m.eps);
            auto rope=[&](float* a,uint32_t heads){for(uint32_t h=0;h<heads;++h){float* z=a+size_t(h)*m.head_dim;for(uint32_t i=0;i<m.head_dim/2;++i){float x0=z[i],x1=z[i+m.head_dim/2];z[i]=x0*cosv[i]-x1*sinv[i];z[i+m.head_dim/2]=x1*cosv[i]+x0*sinv[i];}}};rope(q.data(),m.n_heads);rope(k.data(),m.n_kv);
            for(uint32_t h=0;h<m.n_kv;++h){std::copy_n(k.data()+size_t(h)*m.head_dim,m.head_dim,cache(kc,l,h,pos));std::copy_n(v.data()+size_t(h)*m.head_dim,m.head_dim,cache(vc,l,h,pos));}
            uint32_t ratio=m.n_heads/m.n_kv;
            for(uint32_t h=0;h<m.n_heads;++h){
                uint32_t kh=h/ratio;float* out=att.data()+size_t(h)*m.head_dim;
                float mx=-std::numeric_limits<float>::infinity();uint32_t positive_infinities=0;
                for(uint32_t t=0;t<=pos;++t){
                    const float* kk=cache(kc,l,kh,t);float dot=0;
                    for(uint32_t d=0;d<m.head_dim;++d)dot+=q[size_t(h)*m.head_dim+d]*kk[d];
                    scores[t]=dot*scale;
                    if(scores[t]==std::numeric_limits<float>::infinity())++positive_infinities;
                    else if(std::isfinite(scores[t]))mx=std::max(mx,scores[t]);
                }
                float den=0;
                if(positive_infinities){
                    for(uint32_t t=0;t<=pos;++t){scores[t]=scores[t]==std::numeric_limits<float>::infinity()?1.f:0.f;den+=scores[t];}
                } else if(std::isfinite(mx)) {
                    for(uint32_t t=0;t<=pos;++t){scores[t]=std::isfinite(scores[t])?std::exp(scores[t]-mx):0.f;den+=scores[t];}
                }
                if(!(den>0.f)||!std::isfinite(den)){
                    const float uniform=1.f/float(pos+1);
                    std::fill_n(scores.begin(),pos+1,uniform);
                } else {
                    for(uint32_t t=0;t<=pos;++t)scores[t]/=den;
                }
                std::fill_n(out,m.head_dim,0.f);
                for(uint32_t t=0;t<=pos;++t){float p=scores[t];const float* vv=cache(vc,l,kh,t);for(uint32_t d=0;d<m.head_dim;++d)out[d]+=p*vv[d];}
            }
            matvec(m,w[3],att.data(),xb.data());add(x.data(),xb.data(),m.dim);
            rmsnorm(xb.data(),x.data(),n[3],m.dim,m.eps);matvec(m,w[4],xb.data(),gate.data());matvec(m,w[5],xb.data(),up.data());for(uint32_t i=0;i<m.hidden;++i)gate[i]=silu(gate[i])*up[i];matvec(m,w[6],gate.data(),xb.data());add(x.data(),xb.data(),m.dim);
        }
        if(calc){rmsnorm(x.data(),x.data(),reinterpret_cast<const float*>(m.final_norm),m.dim,m.eps);Matrix e{m.embed,m.vocab,m.dim,m.rs(m.dim)};matvec(m,e,x.data(),logits.data());}
    }
    uint32_t next()const{return uint32_t(std::max_element(logits.begin(),logits.end())-logits.begin());}
};

void run_turn(const Model& m,Engine& e,const std::string& text,uint32_t& pos,uint32_t maxgen){
    auto toks=m.encode(text);std::vector<uint32_t> prompt{kImStart,kUser,kNl};prompt.insert(prompt.end(),toks.begin(),toks.end());prompt.insert(prompt.end(),{kImEnd,kNl,kImStart,kAssistant,kNl,kThinkStart,kNl2,kThinkEnd,kNl2});
    if(uint64_t(pos)+prompt.size()+maxgen+2>=e.maxseq){std::cerr<<"[chat] context full, resetting KV-cache\n";pos=0;}
    std::cerr<<"[chat] ctx="<<pos<<"/"<<e.maxseq<<"\n";
    const auto prefill_start=std::chrono::steady_clock::now();
    for(size_t i=0;i<prompt.size();++i)e.forward(prompt[i],pos++,i+1==prompt.size());
    const auto prefill_end=std::chrono::steady_clock::now();
    const auto decode_start=prefill_end;
    uint32_t token=e.next(),made=0;
    while(made<maxgen&&token!=m.eos1&&token!=m.eos2&&pos<e.maxseq){
        std::string s=m.token(token);std::cout<<s<<std::flush;++made;
        if(made==maxgen||pos+1>=e.maxseq){e.forward(token,pos++,false);break;}
        e.forward(token,pos++,true);token=e.next();
    }
    const auto decode_end=std::chrono::steady_clock::now();
    std::cout<<"\n"<<std::flush;
    if(pos+2<e.maxseq){e.forward(m.eos1,pos++,false);e.forward(kNl,pos++,false);}
    std::cerr<<"[chat] ctx="<<pos<<"/"<<e.maxseq<<"\n";
    const double prefill_s=std::chrono::duration<double>(prefill_end-prefill_start).count();
    const double decode_s=std::chrono::duration<double>(decode_end-decode_start).count();
    std::cerr<<"[speed] prompt="<<prompt.size()<<" tokens, "<<(prefill_s?prompt.size()/prefill_s:0)<<" tok/s; generated="<<made<<" tokens, "<<(decode_s?made/decode_s:0)<<" tok/s\n";
}
}

int main(int argc,char** argv){
    try { if(argc<2){std::cerr<<"usage: asm-llm-arm64 MODEL.asmbin [context=2048] [max_tokens=128]\n";return 2;}
        uint32_t ctx=argc>2?std::stoul(argv[2]):2048,gen=argc>3?std::stoul(argv[3]):128;
        Model m(argv[1]); if(m.vocab!=kVocab)throw std::runtime_error("unexpected vocabulary size"); Engine e(m,ctx);std::cerr<<"asm-llm ARM64 native; Qwen3-0.6B ready ("<<m.file.size()<<" bytes)\nType /clear or /quit.\n";
        uint32_t pos=0;std::string line;while(std::cout<<"asm-llm-arm64> "&&std::getline(std::cin,line)){if(line=="/quit"||line=="/halt")break;if(line=="/clear"||line=="/new"||line=="/reset"){pos=0;std::fill(e.kc.begin(),e.kc.end(),0);std::fill(e.vc.begin(),e.vc.end(),0);std::cout<<"history cleared\n";continue;}if(line.empty())continue;run_turn(m,e,line,pos,gen);}return 0;
    } catch(const std::exception& ex){std::cerr<<"error: "<<ex.what()<<"\n";return 1;}
}
