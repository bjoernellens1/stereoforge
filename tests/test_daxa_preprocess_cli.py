"""CPU-only contract for the actual app's after-timing integer diagnostic."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_actual_stereo_integer_diagnostic_accepts_exact_words_and_rejects_mutations(tmp_path):
    source = (ROOT / "apps/daxa_preprocess_benchmark.cpp").read_text()
    assert "void verify_stereo_integer(" in source
    helper = source[source.index("void verify_stereo_integer("):source.index("json metadata(")]
    cpp = tmp_path / "diagnostic.cpp"
    cpp.write_text(r'''
#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>
struct StereoConfig {uint32_t width=5,height=5,disparities=3,paths=4,p2=32;};
struct DaxaStereoReadback {std::array<std::vector<uint32_t>,2> census,census_valid;std::vector<uint32_t> costs,sums,validity,packed_penalties;};
''' + helper + r'''
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}if(!caught)throw std::runtime_error("mutation was accepted");}
int main(){for(auto bits:{8u,16u}){StereoConfig c;if(bits==16){c.paths=8;c.p2=8191;}DaxaStereoReadback reference,actual;
for(unsigned e=0;e<2;++e){reference.census[e].assign(25,0x123456);reference.census_valid[e].assign(25,1);}reference.validity.assign(25,1);
for(unsigned i=0;i<25;++i){reference.costs.insert(reference.costs.end(),{0,10,20});reference.sums.insert(reference.sums.end(),{0,c.paths*10+(bits==8?4u:256u),c.paths*20+(bits==8?128u:65528u)});}actual=reference;
// Independent explicit words: d0=0,d1=4,d2=128 (8bit); or d1=256,d2=65528 (16bit). Tails0.
for(unsigned i=0;i<25;++i){if(bits==8)actual.packed_penalties.push_back(0x00800400u);else{actual.packed_penalties.push_back(0x01000000u);actual.packed_penalties.push_back(0x0000fff8u);}}
verify_stereo_integer(c,actual,reference,bits);
auto changed=actual;changed.census[0][0]^=1;rejects([&]{verify_stereo_integer(c,changed,reference,bits);});
changed=actual;changed.census_valid[1][0]=0;rejects([&]{verify_stereo_integer(c,changed,reference,bits);});
changed=actual;changed.costs[0]++;rejects([&]{verify_stereo_integer(c,changed,reference,bits);});
changed=actual;changed.sums[0]++;rejects([&]{verify_stereo_integer(c,changed,reference,bits);});
changed=actual;changed.validity[0]=0;rejects([&]{verify_stereo_integer(c,changed,reference,bits);});
changed=actual;changed.packed_penalties.back()^=0x01000000u;rejects([&]{verify_stereo_integer(c,changed,reference,bits);});
}return 0;}
''')
    executable = tmp_path / "diagnostic"
    subprocess.run(["c++", "-std=c++20", "-O2", str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_integer_diagnostic_is_after_timing_and_does_not_claim_compound_float():
    source = (ROOT / "apps/daxa_preprocess_benchmark.cpp").read_text()
    assert 'if(check&&stereo&&options.contains("--dump-prefix"))' in source
    assert source.index('if(check&&stereo&&options.contains("--dump-prefix"))') > source.index('for(uint32_t i=0;i<frames+warmup;')
    assert '"integer_reference_checked",stereo_integer_checked' in source
    assert '"packed_penalties_reference_checked",stereo_packed_checked' in source
    assert '"compound_float_reference_checked",false' in source
