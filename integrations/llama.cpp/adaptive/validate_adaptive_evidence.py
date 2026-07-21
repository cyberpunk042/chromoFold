#!/usr/bin/env python3
import json,sys
p=json.load(open(sys.argv[1]))
def need(cond,msg):
    if not cond: raise SystemExit(msg)
c=p['codec_distribution'];q=p['quality_budget'];r=p['recompression'];pc=p['persistent_cache'];ci=p['cache_integrity'];rt=p['runtime'];mb=p['memory_budget']
need(c['int2_pages']>0 and c['int4_pages']>0 and c['escalation_pages']>0,'mixed codec distribution not proven')
need(c['mixed_codec_attention_launches']>0,'mixed codec attention not proven')
need(q['violations']==0 and q['invalid_candidates_rejected']>0,'quality budget not enforced')
need(r['successes']>0 and r['bytes_saved']>0,'recompression savings not proven')
need(mb['honored'] is True and mb['peak_bytes']<=mb['configured_bytes'],'memory budget exceeded')
need(pc['pages_written']>0 and pc['pages_reloaded']>0 and pc['warm_hits']>0,'persistent warm cache not proven')
need(ci['corrupted_records_rejected']>0,'corruption rejection not proven')
need(rt['dense_fallback_launches']==0 and rt['cuda_errors']==0,'runtime errors or dense fallback')
need(rt['page_refs_at_shutdown']==0 and rt['snapshot_refs_at_shutdown']==0,'shutdown references leaked')
print('M13 adaptive evidence valid')
