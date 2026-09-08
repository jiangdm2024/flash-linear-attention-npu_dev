"""Mega 编译类型分发合同；CPU stub 只检查分派，不代替 NPU 精度。"""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd"
INTERNAL = OP / "op_kernel/internal"
HELPERS = {
    "arch22": INTERNAL / "arch22/operators/chunk_recompute_wu_fwd_ho/op_kernel/chunk_recompute_wu_fwd_ho.cpp",
    "arch35": INTERNAL / "gated_delta_rule_state_update_output/chunk_gated_delta_rule_state_update_output.cpp",
}
H_MARKER = "template <typename InputT, typename TileShapes>\n__aicore__ inline void DispatchFwdH"
O_MARKER = "template <typename InputT>\n__aicore__ inline void DispatchFwdO"


def function_text(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("arch", HELPERS)
def test_entry_uses_generated_input_dtype_for_both_existing_keys(arch):
    source = (INTERNAL / arch / f"chunk_gated_delta_rule_fwd_{arch}.cpp").read_text(encoding="utf-8")
    entry = source.split('extern "C"', 1)[1]
    assert entry.count("GDN::RunPhase6<DTYPE_Q,") == 2
    assert "dtypeMode" not in entry
    for key, value in ((1, 128), (2, 256)):
        assert f"TILING_KEY_IS({key})" in entry
        assert f"GDNFwdHTileShapes{value}" in entry
    assert "DispatchFwdH<InputT, TileShapes>(" in source
    assert "DispatchFwdO<InputT>(" in source


@pytest.mark.parametrize("arch", HELPERS)
def test_mega_h_only_specializes_known_types_and_keeps_state_and_gk(arch):
    helper = HELPERS[arch].read_text(encoding="utf-8")
    dispatch = function_text(helper, H_MARKER)
    assert "hTiling->stateDataType == 2" in dispatch
    assert "hTiling->useGk" in dispatch
    assert "hTiling->dataType" not in dispatch
    assert "hTiling->gDataType" not in dispatch
    for state in ("float", "InputT"):
        for gated in ("true", "false"):
            assert f"RunFwdH<InputT, float, {state}, TileShapes, {gated}>(" in dispatch
    assert dispatch.count("RunFwdH<") == 4
    if arch == "arch22":
        assert "InputT, GT, StateT, float, TileShapes, kGated, true, false, true>" in helper
    else:
        # Preserve the latest A5 chunk-pipeline mode from upstream f6c6e9a;
        # the compatibility branch outside __CCE_AICORE__==310 remains false.
        assert "#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310" in helper
        assert "InputT, GT, StateT, float, TileShapes, kGated, true, false, true>" in helper
        assert "InputT, GT, StateT, float, TileShapes, kGated, true, false, false>" in helper
    assert "DTYPE_FINAL_STATE" not in dispatch
    assert "DTYPE_INITIAL_STATE" not in dispatch


@pytest.mark.parametrize("arch", HELPERS)
def test_mega_o_uses_one_known_input_gate_pair(arch):
    helper = HELPERS[arch].read_text(encoding="utf-8")
    dispatch = function_text(helper, O_MARKER)
    assert dispatch.count("RunFwdO<InputT, float>(") == 1
    assert "if (" not in dispatch
    assert "GDNFwdOKernel<InputT, GT, float, true>" in helper


def test_final_state_placeholder_does_not_determine_initial_state_dtype():
    l2 = (OP / "op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp").read_text(encoding="utf-8")
    placeholder = l2.split("if (!outputFinalState)", 1)[1].split("const aclTensor *gCumsumCompute", 1)[0]
    assert "AllocTensor(MakeShape({1}), DataType::DT_FLOAT" in placeholder
    for arch in HELPERS:
        tiling = (OP / f"op_host/op_tiling/{arch}/chunk_gated_delta_rule_fwd_{arch}_state_output_tiling.cpp").read_text(encoding="utf-8")
        assert "useInitialState ? DtypeToEnum(initialDesc->GetDataType()) : GDN_FWD_H_DTYPE_FP32" in tiling


def test_legacy_arch22_dispatch_is_excluded_from_mega_translation_unit():
    helper = HELPERS["arch22"].read_text(encoding="utf-8")
    guard = "#ifndef GDN_CHUNK_RECOMPUTE_WU_FWD_HO_IMPL_ONLY"
    legacy_h = helper.index("template <typename TileShapes>\n__aicore__ inline void DispatchFwdH")
    assert helper.rfind(guard, 0, legacy_h) > helper.index(H_MARKER)
    legacy_o = helper.index("__aicore__ inline void DispatchFwdO", helper.index(O_MARKER) + len(O_MARKER))
    assert helper.rfind(guard, 0, legacy_o) > helper.index(O_MARKER)


CPP_STUB = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>
#define __aicore__
#define __gm__
using GM_ADDR = unsigned char *;
struct half {};
struct bfloat16_t {};
struct Tile128 {};
struct Tile256 {};
struct ChunkGatedDeltaRuleFwdHTilingData { int stateDataType; bool useGk; };
struct ChunkFwdOTilingData {};
template<class T> constexpr int dtype() {
    if constexpr (std::is_same_v<T, float>) return 2;
    if constexpr (std::is_same_v<T, bfloat16_t>) return 1;
    return 0;
}
struct Record {
    int input, gate, state;
    bool gated;
    std::size_t argc;
    std::array<std::uintptr_t, 13> args;
} seen{};
int calls = 0;
template<class InputT, class GT, class StateT, class TileShapes, bool gated, class... Args>
void RunFwdH(Args... args) {
    static_assert(sizeof...(args) == 13);
    seen = {dtype<InputT>(), dtype<GT>(), dtype<StateT>(), gated, sizeof...(args),
            {reinterpret_cast<std::uintptr_t>(args)...}};
    ++calls;
}
template<class InputT, class GT, class... Args>
void RunFwdO(Args... args) {
    static_assert(sizeof...(args) == 10);
    seen = {dtype<InputT>(), dtype<GT>(), -1, false, sizeof...(args),
            {reinterpret_cast<std::uintptr_t>(args)...}};
    ++calls;
}
// ACTUAL_DISPATCH
template<class InputT, class TileShapes> void check() {
    unsigned char storage[13][64]{};
    std::array<GM_ADDR, 13> args{};
    for (unsigned i = 0; i < args.size(); ++i) args[i] = storage[i];
    ChunkGatedDeltaRuleFwdHTilingData h{};
    args[11] = reinterpret_cast<GM_ADDR>(&h);
    for (int state : {dtype<InputT>(), 2}) {
        for (bool gated : {false, true}) {
            h = {state, gated};
            DispatchFwdH<InputT, TileShapes>(args[0], args[1], args[2], args[3], args[4],
                args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);
            assert(seen.input == dtype<InputT>() && seen.gate == 2);
            assert(seen.state == state && seen.gated == gated && seen.argc == 13);
            for (unsigned i = 0; i < args.size(); ++i)
                assert(seen.args[i] == reinterpret_cast<std::uintptr_t>(args[i]));
        }
    }
    ChunkFwdOTilingData o{};
    DispatchFwdO<InputT>(args[0], args[1], args[2], args[3], args[4], args[5],
        args[6], args[7], args[8], &o);
    assert(seen.input == dtype<InputT>() && seen.gate == 2 && seen.argc == 10);
    for (unsigned i = 0; i < 9; ++i)
        assert(seen.args[i] == reinterpret_cast<std::uintptr_t>(args[i]));
    assert(seen.args[9] == reinterpret_cast<std::uintptr_t>(&o));
}
int main() {
    check<half, Tile128>();
    check<half, Tile256>();
    check<bfloat16_t, Tile128>();
    check<bfloat16_t, Tile256>();
    assert(calls == 20);
    std::cout << "20 typed calls passed\n";
}
"""


@pytest.mark.parametrize("arch", HELPERS)
def test_real_dispatch_bodies_with_host_cpp_stubs(arch, tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("本机无 C++ 编译器；此项必须在构建环境补跑，不替代 NPU 精度")
    helper = HELPERS[arch].read_text(encoding="utf-8")
    functions = function_text(helper, H_MARKER) + "\n" + function_text(helper, O_MARKER)
    source = tmp_path / "dispatch.cpp"
    executable = tmp_path / "dispatch.exe"
    source.write_text(CPP_STUB.replace("// ACTUAL_DISPATCH", functions), encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=120)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
    assert "20 typed calls passed" in result.stdout
