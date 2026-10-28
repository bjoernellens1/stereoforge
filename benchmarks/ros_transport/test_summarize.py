from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parent))
from summarize import summarize


def test_counts_skips_and_duplicates(tmp_path):
    pub=tmp_path/'p.csv'; sub=tmp_path/'s.csv'
    pub.write_text('id,target_ns,begin_ns,depth_wall_ns,depth_cpu_ns,rgb_wall_ns,rgb_cpu_ns\n1,10,12,3,2,4,2\n2,20,0,0,0,0,0\n3,30,31,5,3,6,3\n')
    sub.write_text('id,arrival_ns,kind\n1,15,d\n1,16,d\n3,38,r\n')
    result=summarize(pub,sub)
    assert result['offered']==2 and result['pacing_skips']==1
    assert result['outputs']['depth']['unique_received']==1
    assert result['outputs']['depth']['duplicates']==1
    assert result['outputs']['depth']['offered_not_received']==1
    assert result['outputs']['rgb']['unique_received']==1
    sub.write_text('id,arrival_ns,kind\n2,40,d\n')
    with pytest.raises(ValueError): summarize(pub,sub)
