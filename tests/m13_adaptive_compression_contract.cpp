#include "chromofold/adaptive_compression.h"
#include "chromofold/persistent_page_store.h"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>
int main(){cf_adaptive_config c{CF_POLICY_HYBRID,2,16,32,0.25f,0.99f,1ull<<30};auto*r=cf_adaptive_create(&c);assert(r);std::vector<float>k(256),v(256),dk(256),dv(256);for(size_t i=0;i<k.size();++i){k[i]=std::sin(float(i)*.01f);v[i]=std::cos(float(i)*.02f);}cf_encoded_page p{};assert(cf_adaptive_encode(r,k.data(),v.data(),k.size(),&p)==0);assert(cf_adaptive_decode(&p,dk.data(),dv.data(),dk.size())==0);assert(p.key_codec.codec>=CF_PAGE_INT2_BLOCKWISE);std::filesystem::create_directories("build/m13");cf_persistent_store_config sc{"build/m13/cache.cfp","model",1ull<<30,1,1,1};auto*s=cf_persistent_store_open(&sc);assert(s);cf_persistent_page_key key{0,128,8,128,42};assert(cf_persistent_store_put(s,&key,&p)==0);cf_persistent_store_close(s);sc.write_enabled=0;s=cf_persistent_store_open(&sc);cf_encoded_page loaded{};assert(cf_persistent_store_get(s,&key,&loaded)==0);assert(cf_persistent_store_verify(s)==0);cf_encoded_page_release(&loaded);cf_persistent_store_close(s);cf_encoded_page_release(&p);cf_adaptive_destroy(r);return 0;}
