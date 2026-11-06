// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace {
using namespace Shader;

struct TensorMode {
    std::string previous;
    bool existed{};

    explicit TensorMode(const char* mode) {
        if (const char* value = std::getenv("EDEN_NX2_TENSOR")) {
            previous = value;
            existed = true;
        }
        Set(mode);
    }

    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("EDEN_NX2_TENSOR", value ? value : "");
#else
        if (value) {
            setenv("EDEN_NX2_TENSOR", value, 1);
        } else {
            unsetenv("EDEN_NX2_TENSOR");
        }
#endif
    }

    ~TensorMode() {
        Set(existed ? previous.c_str() : nullptr);
    }
};

struct TensorProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile;

    explicit TensorProgram(u32 count = 1, u32 software_depth = 0) {
        program.stage = Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        program.info.uses_subgroup_shuffles = count != 0 && software_depth != 0;
        program.info.uses_subgroup_invocation_id = count != 0;
        profile.supported_spirv = 0x00010500;
        profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
        profile.max_subgroup_size = 32;
        IR::IREmitter ir{block};
        if (count == 0) {
            return;
        }
        const auto lane = ir.LaneId();
        std::array<IR::U32, 8> inputs;
        for (u32 i = 0; i < inputs.size(); ++i) {
            inputs[i] = software_depth ? ir.IAdd(ir.Imm32(0x3c003c00U + i * 0x01000100U),
                ir.ShiftLeftLogical(lane, ir.Imm32(3))) : ir.IAdd(ir.Imm32((i + 1) * 0x01000100U), lane);
        }
        for (u32 i = 0; i < count; ++i) {
            const auto result = ir.TensorMulAdd(
                                       ir.CompositeConstruct(inputs[0], inputs[1], inputs[2], inputs[3]),
                                       ir.CompositeConstruct(inputs[4], inputs[5]),
                                       ir.CompositeConstruct(inputs[6], inputs[7]), software_depth);
            inputs[6] = IR::U32{ir.CompositeExtract(result, 0)};
            inputs[7] = IR::U32{ir.CompositeExtract(result, 1)};
        }
        if (software_depth) {
            program.info.used_storage_buffer_types = IR::Type::U32;
            program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
            program.info.uses_local_invocation_id = true;
            const auto offset = ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(3));
            block.AppendNewInst(IR::Opcode::WriteStorage32, {ir.Imm32(0), offset, inputs[6]});
            block.AppendNewInst(IR::Opcode::WriteStorage32,
                {ir.Imm32(0), ir.IAdd(offset, ir.Imm32(4)), inputs[7]});
        }
    }

    std::vector<u32> Emit(std::string_view name = {}) {
        auto code = Backend::SPIRV::EmitSPIRV(profile, program);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
            directory && !name.empty()) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
            REQUIRE(file.good());
        }
        return code;
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        REQUIRE(offset + size <= code.size());
        if ((code[offset] & 0xffffU) == opcode) {
            result.emplace_back(code.begin() + offset + 1, code.begin() + offset + size);
        }
        offset += size;
    }
    return result;
}

bool HasCapability(const std::vector<u32>& code, u32 capability) {
    const auto capabilities = Instructions(code, 17);
    return std::ranges::any_of(capabilities, [=](const auto& words) {
        return words[0] == capability;
    });
}

}

TEST_CASE("Software HMMA shares one helper per depth without tensor extensions",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram single{1, depth};
        const auto first = single.Emit(depth == 8 ? "software-k8" : "software-k16");
        REQUIRE_FALSE(HasCapability(first, 5357));
        REQUIRE(Instructions(first, 57).size() == 1); // OpFunctionCall
        TensorProgram repeated{64, depth};
        const auto many = repeated.Emit(depth == 8 ? "software-k8-many" : "software-k16-many");
        REQUIRE(Instructions(many, 57).size() == 64);
        // Three fragment-vector constructions, the call and two result extracts
        // take at most 40 words per site; the matrix body itself is never duplicated.
        REQUIRE(many.size() < first.size() + 64 * 40);
    }
}

TEST_CASE("Software HMMA rounds once per uniform K8 group", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        const auto code = fixture.Emit();
        std::map<u32, u32> constants;
        for (const auto& words : Instructions(code, 43)) { // OpConstant
            constants.emplace(words[1], words[2]);
        }
        // One K=8 group is emitted once; K=16 runs it twice from a uniform counter.
        const auto loops = Instructions(code, 246); // OpLoopMerge
        if (depth == 8) {
            REQUIRE(loops.empty());
        } else {
            REQUIRE(loops.size() == 1);
            const auto limits = Instructions(code, 176); // OpULessThan
            REQUIRE(limits.size() == 1);
            REQUIRE(constants.at(limits.front()[3]) == 2);
            const auto loads = Instructions(code, 61); // OpLoad
            const auto counter_load = std::ranges::find_if(
                loads, [&](const auto& words) { return words[1] == limits.front()[2]; });
            REQUIRE(counter_load != loads.end());
            const u32 counter = (*counter_load)[2];
            u32 counter_stores = 0;
            for (const auto& store : Instructions(code, 62)) { // OpStore
                if (store[0] != counter) {
                    continue;
                }
                ++counter_stores;
                if (constants.contains(store[1])) {
                    REQUIRE(constants.at(store[1]) == 0);
                }
            }
            REQUIRE(counter_stores == 2); // Zero, then the increment: no lane input.
        }
        u32 rounding_operations = 0;
        u32 fma_operations = 0;
        for (const auto& words : Instructions(code, 12)) { // OpExtInst
            rounding_operations += words[3] == 2;          // GLSL.std.450 RoundEven
            fma_operations += words[3] == 50;              // GLSL.std.450 Fma
        }
        REQUIRE(rounding_operations == 4); // One per accumulator, after each K=8 group.
        REQUIRE(fma_operations == 32);
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
        REQUIRE(Instructions(code, 247).empty());      // No per-pair rounding branch.
    }
}

TEST_CASE("Software matrix helpers also emit for 64-lane host subgroups",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        fixture.profile.max_subgroup_size = 64;
        const auto code = fixture.Emit(depth == 8 ? "software-k8-wave64" : "software-k16-wave64");
        REQUIRE_FALSE(HasCapability(code, 5357));
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
    }
}

TEST_CASE("Captured Bananza matrix shaders lower without an environment override",
          "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    for (const Ampere::Instruction instruction : {
             Ampere::Instruction{{0x102c723c, 0x30, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1020723c, 0x24, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1418723c, 0x1c, 0x8ff, 0x04aff000}}}) {
        const std::array code{instruction,
                              Ampere::Instruction{{0x794d, 0, 0x03800000, 0}}};
        const auto lowered = Ampere::LowerToMaxwell(code, false, true);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(Maxwell::Decode(lowered.instructions[0]) == Maxwell::Opcode::TENSOR_MUL_ADD);
    }
}

TEST_CASE("Native matrix lowering selects compact software helpers without legacy toggles", "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    std::array<Ampere::Instruction, 2> code{{
        {{0x0810723c, 12, 0x8ff, 0}}, {{0x0000794d, 0, 0x03800000, 0}}}};
    const auto supported = Ampere::LowerToMaxwell(code, false, false);
    REQUIRE(static_cast<bool>(supported));
    REQUIRE(supported.instructions.size() > 100);
    const auto compact = Ampere::LowerToMaxwell(code, false, true);
    REQUIRE(compact);
    REQUIRE(compact.instructions.size() == 2);
    REQUIRE(Maxwell::Decode(compact.instructions.front()) == Maxwell::Opcode::TENSOR_MUL_ADD);
    REQUIRE((compact.instructions.front() & (1ULL << 48)) != 0);
    for (const char* legacy : {"fast", "stub", "off"}) {
        TensorMode::Set(legacy);
        REQUIRE(Ampere::LowerToMaxwell(code, false, true).instructions == compact.instructions);
        REQUIRE(Ampere::LowerToMaxwell(code).instructions == supported.instructions);
    }
    code[0].words[0] &= ~0x7000U;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[0] |= 0x7000;
    code[0].words[3] = 1;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[3] = 0;
    code[0].words[0] += 1U << 24;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
}

TEST_CASE("Native matrix lowering rejects matrix multiplies when asked", "[video_core][tensor][ampere]") {
    const std::array<Ampere::Instruction, 2> code{{
        {{0x0810723c, 12, 0x8ff, 0}}, {{0x0000794d, 0, 0x03800000, 0}}}};
    const auto rejected = Ampere::LowerToMaxwell(code, {.tensor = true, .reject_matrix = true});
    REQUIRE_FALSE(static_cast<bool>(rejected));
    REQUIRE(rejected.error.find("matrix multiply") != std::string_view::npos);
}
