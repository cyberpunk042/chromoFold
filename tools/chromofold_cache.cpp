#include "chromofold/persistent_page_store.h"
#include <iostream>
#include <string>
int main(int argc,char**argv){if(argc<4){std::cerr<<"usage: chromofold-cache <verify|compact|inspect> <path> <model-sha256>\n";return 2;}cf_persistent_store_config c{argv[2],argv[3],0,1,0,1};cf_persistent_store*s=cf_persistent_store_open(&c);if(!s)return 1;std::string cmd=argv[1];int rc=0;if(cmd=="verify")rc=cf_persistent_store_verify(s);else if(cmd=="compact")rc=cf_persistent_store_compact(s);else if(cmd=="inspect"){cf_persistent_store_counters n{};cf_persistent_store_get_counters(s,&n);std::cout<<"records_loaded="<<n.records_loaded<<" corrupted="<<n.corrupted_records_rejected<<"\n";}else rc=2;cf_persistent_store_close(s);return rc;}
