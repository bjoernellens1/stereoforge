"""CPU checks of the real local memory selector and complete phase accounting."""
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import pytest
from test_kernel_variant_hotpath import body
import compare_kernel_variants as variants

ROOT=Path(__file__).resolve().parents[1]


def test_readback_selector_requires_compatible_cached_coherent_host_memory(tmp_path):
    source=(ROOT/'apps/daxa_variant_benchmark.cpp').read_text()
    selector=body(source,'static uint32_t select_readback_memory_type(')
    program='''#include <vulkan/vulkan.h>
#include <string_view>
#include <stdexcept>
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static uint32_t select_readback_memory_type(uint32_t compatible,VkPhysicalDeviceMemoryProperties const& memory,std::string_view policy){'''+selector+'''}
int main(){
    VkPhysicalDeviceMemoryProperties m{};m.memoryTypeCount=4;
    m.memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    m.memoryTypes[1].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    m.memoryTypes[2].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    m.memoryTypes[3].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if(select_readback_memory_type(15,m,"first-coherent")!=1)return 1;
    if(select_readback_memory_type(15,m,"cached-coherent")!=3)return 2;
    for(auto policy:{"cached-coherent","unknown"}){bool rejected=false;try{select_readback_memory_type(7,m,policy);}catch(...){rejected=true;}if(!rejected)return 3;}
    bool rejected=false;try{select_readback_memory_type(1,m,"first-coherent");}catch(...){rejected=true;}return rejected?0:4;
}'''
    code=tmp_path/'selector.cpp';code.write_text(program);binary=tmp_path/'selector'
    subprocess.run(['c++','-std=c++20',str(code),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_exact_cpu_phases_include_zero_stages_and_reject_missing_or_unaccounted_time():
    row=dict(input_stage=0.0,enqueue=1.0,completion_wait=2.0,readback_copy=0.0,query=0.5,total=3.5)
    variants.validate_phase_rows([row],1)
    for broken in ({**row,'total':3.6},{**row,'query':-1},{**row,'query':float('nan')},{key:value for key,value in row.items() if key!='query'}):
        with pytest.raises(ValueError):variants.validate_phase_rows([broken],1)


def test_memory_policy_type_and_flags_are_part_of_specialization():
    first=variants.memory_specialization(dict(readback_memory_policy='first-coherent',readback_memory_type_index=2,readback_memory_property_flags=6))
    cached=variants.memory_specialization(dict(readback_memory_policy='cached-coherent',readback_memory_type_index=5,readback_memory_property_flags=14))
    assert first!=cached
    with pytest.raises(ValueError):variants.memory_specialization(dict(readback_memory_policy='cached-coherent',readback_memory_type_index=2,readback_memory_property_flags=6))


def test_comparison_rejects_different_actual_memory_types_even_with_same_requested_policy():
    import json
    from test_kernel_variants import identity
    first=identity();second=identity()
    specialization=json.loads(second['specialization']);specialization['readback_memory']['type_index']=3
    second['specialization']=json.dumps(specialization)
    with pytest.raises(ValueError):variants.validate_comparable(first,second)


@pytest.mark.parametrize('flags',[-1,2**32])
def test_memory_flags_are_unsigned_32_bit_properties(flags):
    with pytest.raises(ValueError):variants.memory_specialization(dict(readback_memory_policy='cached-coherent',readback_memory_type_index=5,readback_memory_property_flags=flags))


def test_native_phase_formatter_preserves_zero_values_as_valid_json(tmp_path):
    import json
    source=(ROOT/'apps/daxa_variant_benchmark.cpp').read_text()
    formatter=body(source,'static void phase_rows(')
    sample=body(source,'struct Sample')
    program='#include <iostream>\n#include <vector>\nstruct Sample{'+sample+'};\nstatic void phase_rows(std::ostream& out,std::vector<Sample> const& rows){'+formatter+'}\nint main(){phase_rows(std::cout,{{0,1,2,0,0.5,3,3.5}});}'
    code=tmp_path/'formatter.cpp';code.write_text(program);binary=tmp_path/'formatter'
    subprocess.run(['c++','-std=c++20',str(code),'-o',str(binary)],check=True)
    rows=json.loads(subprocess.check_output([str(binary)],text=True))
    assert rows[0]['input_stage']==0 and rows[0]['readback_copy']==0
    variants.validate_phase_rows(rows,1)


def test_incomplete_or_nonnumeric_wall_totals_fail_closed():
    row=dict(input_stage=0.0,enqueue=1.0,completion_wait=2.0,readback_copy=0.0,query=0.5,total=3.5)
    for totals in ([],[None],[float('nan')]):
        with pytest.raises(ValueError):variants.validate_phase_rows([row],1,totals)


def test_native_options_keep_first_coherent_default_and_bound_experiment_before_device(tmp_path):
    source=(ROOT/'apps/daxa_variant_benchmark.cpp').read_text()
    option_struct=body(source,'struct Options')
    parse=body(source,'static Options options(')
    numeric=body(source,'static uint32_t number(')
    program='''#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
namespace fs=std::filesystem;
struct Options{'''+option_struct+'''};
static uint32_t number(std::string const& s){'''+numeric+'''}
static Options options(int argc,char** argv){'''+parse+'''}
Options call(std::vector<std::string> args){std::vector<char*> ptrs;for(auto& s:args)ptrs.push_back(s.data());return options(int(ptrs.size()),ptrs.data());}
int main(){if(call({"app","--output","unused"}).readback_memory!="first-coherent")return 1;
if(call({"app","--output","unused","--readback-memory","cached-coherent"}).readback_memory!="cached-coherent")return 2;
for(auto args:{std::vector<std::string>{"--frames","301"},{"--frames","29"},{"--warmup","31"},{"--warmup","2"},{"--width","1921","--height","1080"},{"--readback-memory","unknown"}}){
std::vector<std::string> full{"app","--output","unused"};full.insert(full.end(),args.begin(),args.end());bool rejected=false;try{call(full);}catch(...){rejected=true;}if(!rejected)return 3;}return 0;}
'''
    code=tmp_path/'options.cpp';code.write_text(program);binary=tmp_path/'options'
    subprocess.run(['c++','-std=c++20',str(code),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
