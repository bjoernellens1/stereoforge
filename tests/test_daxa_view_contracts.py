from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_shared_projection_and_thin_wrapper_exist():
    ops = ROOT / 'python/stereoforge/shaders/ops/projection.slang'
    wrapper = ROOT / 'python/stereoforge/shaders/daxa/view.slang'
    assert ops.is_file(), 'Daxa compositor shared operations missing'
    text = ops.read_text()
    assert 'bool project_source' in text and 'void gather_project_metadata' in text
    assert 'StructuredBuffer<' not in text and 'vk::' not in text
    assert '#include "../ops/projection.slang"' in wrapper.read_text()

def test_daxa_view_runtime_has_no_rhi_owner():
    text = (ROOT / 'src/daxa_view.cpp').read_text()
    assert 'rhi::' not in text and 'NativeGeometry' not in text
    assert 'latest_queue_submit_index' in text

def test_offline_spirv_parameter_layout():
    """Read actual emitted SPIR-V offsets, independent of compiler reflection JSON."""
    import struct
    import pytest
    path = ROOT / 'build/daxa/daxa-view-resolve_layers.spv'
    if not path.is_file():
        pytest.skip('Daxa offline shader has not been built')
    data = path.read_bytes()
    words = struct.unpack(f'<{len(data)//4}I', data)
    names, decorations = {}, []
    i = 5
    while i < len(words):
        length, opcode = words[i] >> 16, words[i] & 65535
        assert length
        args = words[i+1:i+length]
        if opcode == 5:  # OpName
            names[args[0]] = struct.pack(f'<{len(args)-1}I', *args[1:]).split(b'\0')[0].decode()
        if opcode == 72:  # OpMemberDecorate
            decorations.append(args)
        i += length
    table = next(key for key, name in names.items() if name == 'ViewParameters_natural')
    offsets = {d[1]: d[3] for d in decorations if d[0] == table and d[2] == 35}
    assert offsets == dict(enumerate([*range(0, 200, 8), *range(200, 240, 4), 240, 304, 368, 372]))

def assert_legacy_operations_preserved(ops, original):
    import re
    # Expand only the reviewed legacy defaults. Unknown reader/index/bound
    # changes must fail, even if a default is unused in this specialization.
    defaults = """#ifndef SF_METRIC_RGB
#define SF_METRIC_RGB(index) source_rgb[index]
#define SF_PREVIEW_RGB(index) source_rgb[index]
#define SF_PREVIEW_BOUND (2*src_width*src_height)
#define SF_PREVIEW_OWNER(index) index
#endif"""
    assert ops.count(defaults) == 1, 'legacy reader/index/bounds defaults changed'
    ops = ops.replace(defaults, '', 1)
    typed_only = """#ifdef SF_TYPED_VIEW
            if(index!=0xffffffff&&(index>=SF_PREVIEW_BOUND||index/(SF_PREVIEW_BOUND/2)!=source-1))index=0xffffffff;
#endif"""
    assert ops.count(typed_only) == 1, 'unrecognized typed-only conditional'
    ops = ops.replace(typed_only, '', 1)
    for name in ('SF_METRIC_RGB', 'SF_PREVIEW_RGB'):
        ops = re.sub(name+r'\(([^()]*)\)', lambda m: 'source_rgb['+m[1]+']', ops)
    ops = re.sub(r'SF_PREVIEW_OWNER\(([^()]*)\)', lambda m: m[1], ops)
    assert 'SF_' not in ops, 'unexpanded reader or conditional remains'
    stripped = re.sub(r'^(?:(?:RW)?StructuredBuffer<[^>]+> \w+;|uniform \w+ \w+;)\n', '', original, flags=re.M)
    def tokens(text):
        text = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', '', text)
        return re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|<<=|>>=|==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s]', text)
    # Compare every legacy operation, literal, string and control-flow token;
    # formatting and comments are the only ignored differences.
    assert tokens(ops) == tokens(stripped), 'legacy projection/compositor operations changed'


def test_operations_preserve_frozen_algorithm():
    original = (ROOT/'python/stereoforge/shaders/fusion/project.slang').read_text()
    ops = (ROOT/'python/stereoforge/shaders/ops/projection.slang').read_text()
    assert_legacy_operations_preserved(ops, original)


def test_legacy_normalizer_rejects_changed_readers_bounds_projection_and_resolve():
    import pytest
    original = (ROOT/'python/stereoforge/shaders/fusion/project.slang').read_text()
    ops = (ROOT/'python/stereoforge/shaders/ops/projection.slang').read_text()
    for before, after in (
        ('#define SF_METRIC_RGB(index) source_rgb[index]', '#define SF_METRIC_RGB(index) source_confidence[index]'),
        ('#define SF_PREVIEW_RGB(index) source_rgb[index]', '#define SF_PREVIEW_RGB(index) source_rgb[index+1]'),
        ('#define SF_PREVIEW_BOUND (2*src_width*src_height)', '#define SF_PREVIEW_BOUND (src_width*src_height)'),
        ('#define SF_PREVIEW_OWNER(index) index', '#define SF_PREVIEW_OWNER(index) (index+1)'),
        ('#ifdef SF_TYPED_VIEW', '#ifdef UNKNOWN_CONDITIONAL'),
        ('if (local.z<=0) return false;', 'if (local.z<0) return false;'),
        ('output_rgb[target]=SF_METRIC_RGB(index);', 'output_rgb[target]=SF_METRIC_RGB(index+1);'),
        ('output_valid[target]=1;', 'output_valid[target]=0;'),
    ):
        assert before in ops
        with pytest.raises(AssertionError):
            assert_legacy_operations_preserved(ops.replace(before, after, 1), original)


def test_cli_rejects_before_device_creation(tmp_path):
    import subprocess
    import pytest
    binary = ROOT/'build/daxa/daxa_view_benchmark'
    if not binary.is_file():
        pytest.skip('Daxa benchmark has not been built')
    for args, message in [(['--frames', '0'], 'frames must be positive'),
                          (['--mode', 'unknown'], 'mode must be strict or preview'),
                          (['--input', 'host'], 'input must be resident')]:
        result = subprocess.run([str(binary), *args], capture_output=True, text=True)
        assert result.returncode == 2 and message in result.stderr
    (tmp_path/'header.txt').write_text('invalid')
    result = subprocess.run([str(binary), '--fixture', str(tmp_path), '--output', str(tmp_path/'report.json')], capture_output=True, text=True)
    assert result.returncode == 2 and 'invalid fixture header' in result.stderr
    assert not (tmp_path/'report.json').exists()
