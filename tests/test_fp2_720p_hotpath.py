"""CPU-only 720p debug-harness admission/dispatch guards; no GPU timing."""
from pathlib import Path
import re
import pytest
import test_fp2_census_hotpath as chain

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_720p_fixture.cpp'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def guard(s,owner):
    cmd=chain.block(s)
    host='.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    gpu='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    finish='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'
    a=cmd.index('recorder.set_pipeline(census)');b=cmd.index('recorder.set_pipeline(producer)');c=cmd.index('for(uint32_t pass=0;pass<4;++pass)');d=cmd.index('recorder.set_pipeline(recurrence)')
    second=cmd.index('recorder.dispatch({(f.w*f.h+63)/64,1,1})',cmd.index('CensusPush{d.device_address(census1).value()}'))
    if not(host in cmd[:a] and gpu in cmd[second:b] and gpu in cmd[b:c] and gpu in cmd[c:d] and finish in cmd[d:]):raise ValueError('barrier placement')
    for item in ('CensusPush{d.device_address(census0).value()}','CensusPush{d.device_address(census1).value()}',
                 'CostPush{d.device_address(cparam).value()}','Push{d.device_address(parameter).value(),pass}',
                 'd.submit_commands','d.wait_idle()'):
        if item not in cmd:raise ValueError('dispatch invariant '+item)
    if cmd.count('recorder.dispatch({(f.w*f.h+63)/64,1,1})')!=3 or s.count('d.wait_idle()')!=1:raise ValueError('dispatch/wait count')
    for token in ('buffer_host_address','memcpy','memset','create_memory','create_buffer','wait_on_submit','collect_garbage'):
        if token in cmd:raise ValueError('unexpected stage operation '+token)
    required=['w==1280&&h==720', 'f.offsets.back()<=uint64_t(n)*128&&f.offsets.back()<=65000000',
              'daxa_dvc_get_vk_physical_device(context.device.get())',
              'VK_EXT_MEMORY_BUDGET_EXTENSION_NAME','limits.maxMemoryAllocationSize',
              "n<=1'100'000'000ull", "total<=2'000'000'000ull-n",
              "budget_data.heapBudget[i]-budget_data.heapUsage[i]>=heaps[i]+512'000'000ull",
              "uint64_t(info.freeram)*info.mem_unit>=total+512'000'000ull",
              'admission(context,sizes);','validate_regions(regions)',
              '.byte_code_size=uint32_t(sw.size())','.byte_code_size=uint32_t(cw.size())','.byte_code_size=uint32_t(rw.size())',
              'std::vector<char> chunk(8*1024*1024)', 'auto n=std::min<uint64_t>(bytes,chunk.size())']
    for token in required:
        if token not in s:raise ValueError('admission/resource invariant '+token)
    if s.index('admission(context,sizes);')>s.index('auto lo=alloc('):raise ValueError('admission must precede buffer allocation')
    # Pin the admitted byte multiset AND each matching allocation request.
    # Scalar aliases below reflect validated vector lengths at the input boundary.
    sizes='std::vector<size_t> sizes={n*4,n*4,(n+1)*8,size_t(words)*4,size_t(words)*4,size_t(words)*16,size_t(f.scratch_words)*4,f.line_offsets.size()*8,n*4,n*4,n*4,n*4,n*4,n*4,n*4,n*4,sizeof(CensusParams),sizeof(CensusParams),sizeof(Parameters),sizeof(CostParams)}'
    if sizes not in s or len(re.findall(r'\balloc\s*\(',s)) != 20: raise ValueError('admission byte multiset')
    for request,count in {
        'alloc(f.lo.size()*4,f.lo.data())':1, 'alloc(f.hi.size()*4,f.hi.data())':1,
        'alloc(f.offsets.size()*8,f.offsets.data())':1, 'alloc(size_t(words)*4,nullptr)':2,
        'alloc(size_t(words)*16,nullptr)':1, 'alloc(f.scratch_words*4,nullptr)':1,
        'alloc(f.line_offsets.size()*8,f.line_offsets.data())':1, 'alloc(n*4,nullptr)':4,
        'alloc(left.luma.size()*4,left.luma.data())':1, 'alloc(right.luma.size()*4,right.luma.data())':1,
        'alloc(left.support.size()*4,left.support.data())':1, 'alloc(right.support.size()*4,right.support.data())':1,
        'alloc(sizeof(c0),&c0)':1, 'alloc(sizeof(c1),&c1)':1, 'alloc(sizeof(p),&p)':1, 'alloc(sizeof(cp),&cp)':1,
    }.items():
        if s.count(request)!=count: raise ValueError('allocation multiset '+request)
    tail=s[s.index('d.wait_idle();'):]
    if not(tail.index('compare_file(dir/"sums"')<tail.index('compare_file(dir/"paths"')<tail.index('owner.release();context.close()')):raise ValueError('readback/lifetime order')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('owner cleanup')


def test_source():guard(APP.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
 ('w==1280&&h==720','w==1280'),
 ('f.offsets.back()<=uint64_t(n)*128&&f.offsets.back()<=65000000','true'),
 ('daxa_dvc_get_vk_physical_device(context.device.get())','some_other_device'),
 ('VK_EXT_MEMORY_BUDGET_EXTENSION_NAME','"unverified"'),
 ("total<=2'000'000'000ull-n",'true'),
 ("budget_data.heapBudget[i]-budget_data.heapUsage[i]>=heaps[i]+512'000'000ull",'true'),
 ("uint64_t(info.freeram)*info.mem_unit>=total+512'000'000ull",'true'),
 ('admission(context,sizes);',''),('validate_regions(regions)',''),
 ('std::vector<char> chunk(8*1024*1024)','std::vector<char> chunk(bytes)'),
 ('.byte_code_size=uint32_t(sw.size())','.byte_code_size=uint32_t(sw.size()*4)'),
 ('recorder.set_pipeline(producer)','d.wait_idle();recorder.set_pipeline(producer)'),
 ('recorder.set_pipeline(recurrence)','d.buffer_host_address(cost);recorder.set_pipeline(recurrence)'),
 ('owner.release();context.close()','context.close();owner.release()'),
])
def test_mutation(old,new):
    s=APP.read_text();assert old in s
    with pytest.raises((ValueError,IndexError)):guard(s.replace(old,new,1),OWNER.read_text())


@pytest.mark.parametrize('start,end',[
 ('CensusPush{d.device_address(census1).value()}','recorder.set_pipeline(producer)'),
 ('recorder.set_pipeline(producer)','for(uint32_t pass=0;pass<4;++pass)'),
 ('if(pass)recorder.pipeline_barrier','recorder.set_pipeline(recurrence)'),
])
def test_scoped_barrier_mutation(start,end):
    s=APP.read_text();a=s.index(start);b=s.index(end,a);old='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE';assert old in s[a:b]
    with pytest.raises(ValueError):guard(s[:a]+s[a:b].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+s[b:],OWNER.read_text())


def test_fake_actual_source(tmp_path,monkeypatch):
    # Reuse the identical fake recorder but extract THIS fixture's actual block.
    monkeypatch.setattr(chain,'APP',APP)
    chain.test_fake_actual_source(tmp_path)


def test_missing_path_admission_rejected():
    source=APP.read_text()
    old='size_t(words)*4,size_t(words)*4,size_t(words)*16,size_t(f.scratch_words)*4'
    assert old in source
    with pytest.raises(ValueError):guard(source.replace(old,'size_t(words)*4,size_t(words)*4,size_t(f.scratch_words)*4',1),OWNER.read_text())


def test_unadmitted_allocation_rejected():
    source=APP.read_text()
    old='auto sums=alloc(size_t(words)*4,nullptr)'
    assert old in source
    with pytest.raises(ValueError):guard(source.replace(old,'auto extra=alloc(7,nullptr);auto sums=alloc(size_t(words)*4,nullptr)',1),OWNER.read_text())
