#include "chromofold/adaptive_compression.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct cf_adaptive_runtime { cf_adaptive_config config{}; cf_adaptive_counters counters{}; std::string error; };
namespace {
uint64_t checksum64(const uint8_t * data, size_t size) { uint64_t h=1469598103934665603ull; for(size_t i=0;i<size;++i){h^=data[i];h*=1099511628211ull;} return h; }
int bits_for(uint32_t codec){ return codec==CF_PAGE_INT2_BLOCKWISE?2:codec==CF_PAGE_INT4_BLOCKWISE?4:codec==CF_PAGE_INT8_BLOCKWISE?8:16; }
float qmax(uint32_t codec){ return codec==CF_PAGE_INT2_BLOCKWISE?1.0f:codec==CF_PAGE_INT4_BLOCKWISE?7.0f:codec==CF_PAGE_INT8_BLOCKWISE?127.0f:1.0f; }
uint64_t packed_bytes(uint32_t codec,uint32_t n){ if(codec==CF_PAGE_INT2_BLOCKWISE)return (n+3)/4; if(codec==CF_PAGE_INT4_BLOCKWISE)return (n+1)/2; if(codec==CF_PAGE_INT8_BLOCKWISE)return n; return uint64_t(n)*2; }
uint32_t choose(const cf_adaptive_runtime * r,const cf_page_analysis & a){
 const uint32_t candidates[]={CF_PAGE_INT2_BLOCKWISE,CF_PAGE_INT4_BLOCKWISE,CF_PAGE_INT8_BLOCKWISE,CF_PAGE_FP16_RAW};
 for(uint32_t c:candidates){int b=bits_for(c); if(b<(int)r->config.min_bits||b>(int)r->config.max_bits)continue; float e=c==2?a.estimated_int2_error:c==4?a.estimated_int4_error:c==8?a.estimated_int8_error:0; if(e<=r->config.max_page_error)return c;} return CF_PAGE_FP16_RAW;
}
void pack(const float * in,uint32_t n,uint32_t codec,uint32_t block,std::vector<uint8_t>& payload,std::vector<float>& scales){
 payload.assign(packed_bytes(codec,n),0); scales.resize((n+block-1)/block);
 for(uint32_t bi=0;bi<scales.size();++bi){uint32_t s=bi*block,e=std::min(n,s+block);float m=0;for(uint32_t i=s;i<e;++i)m=std::max(m,std::fabs(in[i]));float sc=m?m/qmax(codec):1;scales[bi]=sc;for(uint32_t i=s;i<e;++i){if(codec==16){uint16_t h=(uint16_t)std::lrint(std::clamp(in[i],-65504.0f,65504.0f));std::memcpy(payload.data()+i*2,&h,2);continue;}int q=(int)std::lrint(in[i]/sc);if(codec==2){q=std::clamp(q,-2,1);payload[i/4]|=(uint8_t)(q&3)<<((i%4)*2);}else if(codec==4){q=std::clamp(q,-8,7);payload[i/2]|=(uint8_t)(q&15)<<((i%2)*4);}else payload[i]=(uint8_t)(int8_t)std::clamp(q,-128,127);}}
 }
}
void unpack(const uint8_t * p,const float * scales,uint32_t n,uint32_t codec,uint32_t block,float * out){for(uint32_t i=0;i<n;++i){if(codec==16){uint16_t h;std::memcpy(&h,p+i*2,2);out[i]=(float)(int16_t)h;continue;}int q;if(codec==2){q=(p[i/4]>>((i%4)*2))&3;q=q>=2?q-4:q;}else if(codec==4){q=(p[i/2]>>((i%2)*4))&15;q=q>=8?q-16:q;}else q=(int8_t)p[i];out[i]=q*scales[i/block];}}
int encode_one(const float * in,uint32_t n,uint32_t codec,uint32_t block,uint8_t ** payload,float ** scales,uint64_t * bytes){std::vector<uint8_t> p;std::vector<float>s;pack(in,n,codec,block,p,s);*payload=new uint8_t[p.size()];*scales=new float[s.size()];std::copy(p.begin(),p.end(),*payload);std::copy(s.begin(),s.end(),*scales);*bytes=p.size();return 0;}
}
extern "C" cf_adaptive_runtime * cf_adaptive_create(const cf_adaptive_config * c){if(!c||!c->block_size||c->min_bits>c->max_bits)return nullptr;auto*r=new cf_adaptive_runtime;r->config=*c;return r;}
extern "C" void cf_adaptive_destroy(cf_adaptive_runtime*r){delete r;}
extern "C" int cf_adaptive_analyze(const float*v,uint32_t n,cf_page_analysis*out){if(!v||!n||!out)return-1;double sum=0,sq=0;float m=0;uint64_t o=0;for(uint32_t i=0;i<n;++i){if(std::isnan(v[i]))out->nan_values++;if(std::isinf(v[i]))out->inf_values++;m=std::max(m,std::fabs(v[i]));sum+=v[i];sq+=double(v[i])*v[i];}double mean=sum/n;out->abs_max=m;out->variance=float(sq/n-mean*mean);for(uint32_t i=0;i<n;++i)if(std::fabs(v[i])>m*.75f)o++;out->outlier_ratio=float(o)/n;out->estimated_int2_error=m/3;out->estimated_int4_error=m/15;out->estimated_int8_error=m/255;return(out->nan_values||out->inf_values)?-1:0;}
extern "C" int cf_adaptive_encode(cf_adaptive_runtime*r,const float*k,const float*v,uint32_t n,cf_encoded_page*out){if(!r||!k||!v||!n||!out)return-1;cf_page_analysis ka{},va{};if(cf_adaptive_analyze(k,n,&ka)||cf_adaptive_analyze(v,n,&va))return-1;uint32_t kc=r->config.policy==CF_POLICY_FIXED_INT4?4:choose(r,ka),vc=r->config.policy==CF_POLICY_FIXED_INT4?4:choose(r,va);out->key_codec={1,kc,r->config.block_size,0};out->value_codec={1,vc,r->config.block_size,0};encode_one(k,n,kc,r->config.block_size,&out->key_payload,&out->key_scales,&out->key_payload_bytes);encode_one(v,n,vc,r->config.block_size,&out->value_payload,&out->value_scales,&out->value_payload_bytes);out->value_count=n;out->checksum=checksum64(out->key_payload,out->key_payload_bytes)^checksum64(out->value_payload,out->value_payload_bytes);for(uint32_t c:{kc,vc}){if(c==2)r->counters.int2_pages++;else if(c==4)r->counters.int4_pages++;else if(c==8)r->counters.int8_pages++;else r->counters.fp16_pages++;}r->counters.bytes_saved+=uint64_t(n)*8-out->key_payload_bytes-out->value_payload_bytes;return 0;}
extern "C" int cf_adaptive_decode(const cf_encoded_page*p,float*k,float*v,uint32_t n){if(!p||!k||!v||n!=p->value_count)return-1;unpack(p->key_payload,p->key_scales,n,p->key_codec.codec,p->key_codec.block_size,k);unpack(p->value_payload,p->value_scales,n,p->value_codec.codec,p->value_codec.block_size,v);return 0;}
extern "C" int cf_adaptive_recompress(cf_adaptive_runtime*r,const cf_encoded_page*s,uint32_t codec,cf_encoded_page*out){if(!r||!s||!out)return-1;r->counters.recompression_attempts++;std::vector<float>k(s->value_count),v(s->value_count);if(cf_adaptive_decode(s,k.data(),v.data(),s->value_count))return-1;auto old=r->config;r->config.policy=CF_POLICY_QUALITY_BUDGET;r->config.min_bits=r->config.max_bits=bits_for(codec);int rc=cf_adaptive_encode(r,k.data(),v.data(),s->value_count,out);r->config=old;if(!rc)r->counters.recompression_successes++;return rc;}
extern "C" void cf_encoded_page_release(cf_encoded_page*p){if(!p)return;delete[]p->key_payload;delete[]p->value_payload;delete[]p->key_scales;delete[]p->value_scales;*p={};}
extern "C" int cf_adaptive_get_counters(const cf_adaptive_runtime*r,cf_adaptive_counters*out){if(!r||!out)return-1;*out=r->counters;return 0;}
extern "C" const char * cf_adaptive_last_error(const cf_adaptive_runtime*r){return r?r->error.c_str():"runtime is null";}
