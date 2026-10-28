"""Compile the actual new C++ rank helper without vendor JSON/build dependencies."""
from pathlib import Path
import subprocess


def test_actual_percentile_helper(tmp_path):
    source = (Path(__file__).parents[1] / 'apps/daxa_refinement_common.hpp').read_text()
    begin = source.index('inline size_t percentile_index(')
    body = source.index('{', begin)
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    helper = source[begin:end]
    program = tmp_path / 'rank.cpp'
    program.write_text('#include <cstddef>\n#include <stdexcept>\n#include <cassert>\n' + helper + '''
int main(){
 assert(percentile_index(100,99)==98);assert(percentile_index(100,95)==94);
 assert(percentile_index(100,50)==49);assert(percentile_index(30,99)==29);
 assert(percentile_index(30,95)==28);assert(percentile_index(30,50)==14);
 assert(percentile_index(1,99)==0);assert(percentile_index(300,99)==296);
 bool failed=false;try{percentile_index(0,99);}catch(std::invalid_argument const&){failed=true;}assert(failed);
}
''')
    binary = tmp_path / 'rank'
    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', str(program), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
