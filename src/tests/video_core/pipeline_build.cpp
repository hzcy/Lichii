// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_prebuild.h"

namespace {
using Vulkan::EmissionBudget;
using Vulkan::PipelineBuild;

bool Pending(std::future<void>& result) {
    return result.wait_for(std::chrono::seconds{0}) != std::future_status::ready;
}
} // namespace

TEST_CASE("Production pipeline build holds a draw without blocking its producer",
          "[video_core][async_draw][pipeline_build]") {
    Common::ThreadWorker workers{1, "PipelineTest"};
    EmissionBudget budget{1};
    std::promise<void> release, entered;
    auto gate = release.get_future().share();
    int output = 0;
    PipelineBuild build{&workers, [reservation = std::shared_ptr{budget.TryAcquire(true)},
                                   gate, &entered, &output] {
                            entered.set_value();
                            gate.wait();
                            output = 42;
                        }};
    entered.get_future().wait();
    const bool building = !build.IsBuilt();
    const bool saturated = !budget.TryAcquire(true);
    auto draw = std::async(std::launch::async, [&] { build.Wait(); });
    const bool held = Pending(draw);
    int producer_progress = 0;
    ++producer_progress;
    release.set_value();
    draw.get();
    REQUIRE(building);
    REQUIRE(saturated);
    REQUIRE(held);
    REQUIRE(producer_progress == 1);
    REQUIRE(output == 42);
    REQUIRE(build.IsBuilt());
    REQUIRE_FALSE(build.Failed());
    REQUIRE(budget.Pending() == 0);
    REQUIRE(budget.TryAcquire(true));
}

TEST_CASE("Out of order pipeline completions preserve scheduler draw order",
          "[video_core][async_draw][pipeline_build]") {
    Common::ThreadWorker workers{2, "PipelineTest"};
    std::promise<void> release_first, release_second;
    auto first_gate = release_first.get_future().share();
    auto second_gate = release_second.get_future().share();
    PipelineBuild first{&workers, [first_gate] { first_gate.wait(); }};
    PipelineBuild second{&workers, [second_gate] { second_gate.wait(); }};
    std::vector<int> draws;
    auto scheduler = std::async(std::launch::async, [&] {
        first.Wait();
        draws.push_back(1);
        second.Wait();
        draws.push_back(2);
    });
    release_second.set_value();
    second.Wait();
    const bool first_pending = !first.IsBuilt();
    const bool scheduler_held = Pending(scheduler);
    release_first.set_value();
    scheduler.get();
    REQUIRE(first_pending);
    REQUIRE(scheduler_held);
    REQUIRE(draws == std::vector<int>{1, 2});
}

TEST_CASE("Pipeline worker exceptions release captures and notify every draw waiter",
          "[video_core][async_draw][pipeline_build]") {
    for (const bool standard_exception : {false, true}) {
        Common::ThreadWorker workers{1, "PipelineTest"};
        EmissionBudget budget{1};
        std::promise<void> release, entered;
        auto gate = release.get_future().share();
        PipelineBuild build{&workers, [reservation = std::shared_ptr{budget.TryAcquire(true)},
                                       gate, &entered, standard_exception] {
                                entered.set_value();
                                gate.wait();
                                if (standard_exception) {
                                    throw std::runtime_error("emission failed");
                                }
                                throw 42;
                            }};
        entered.get_future().wait();
        auto first = std::async(std::launch::async, [&] { build.Wait(); });
        auto second = std::async(std::launch::async, [&] { build.Wait(); });
        release.set_value();
        first.get();
        second.get();
        REQUIRE(build.IsBuilt());
        REQUIRE(build.Failed());
        REQUIRE(budget.Pending() == 0);
        REQUIRE(budget.TryAcquire(true));
    }
}

TEST_CASE("Eviction after an inline build leaves a safe queued callback",
          "[video_core][async_draw][pipeline_build]") {
    Common::ThreadWorker workers{1, "PipelineTest"};
    std::promise<void> release, entered;
    auto gate = release.get_future().share();
    workers.QueueWork([gate, &entered] {
        entered.set_value();
        gate.wait();
    });
    entered.get_future().wait();
    int builds = 0;
    auto build = std::make_unique<PipelineBuild>(&workers, [&] { ++builds; });
    build->Wait(true);
    const bool evictable = build->IsBuilt();
    build.reset();
    release.set_value();
    workers.WaitForRequests();
    REQUIRE(evictable);
    REQUIRE(builds == 1);
}

TEST_CASE("Pipeline destruction drains queued work before owner resources disappear",
          "[video_core][async_draw][pipeline_build]") {
    Common::ThreadWorker workers{1, "PipelineTest"};
    EmissionBudget budget{1};
    std::promise<void> release, entered;
    auto gate = release.get_future().share();
    workers.QueueWork([gate, &entered] {
        entered.set_value();
        gate.wait();
    });
    entered.get_future().wait();
    auto resource = std::make_unique<int>(0);
    auto build = std::make_unique<PipelineBuild>(
        &workers, [reservation = std::shared_ptr{budget.TryAcquire(true)},
                   output = resource.get()] { *output = 42; });
    build.reset();
    const int result = *resource;
    resource.reset();
    release.set_value();
    workers.WaitForRequests();
    REQUIRE(result == 42);
    REQUIRE(budget.Pending() == 0);
}

TEST_CASE("Pipeline destruction waits for a running callback and its cleanup",
          "[video_core][async_draw][pipeline_build]") {
    Common::ThreadWorker workers{1, "PipelineTest"};
    EmissionBudget budget{1};
    std::promise<void> release, entered;
    auto gate = release.get_future().share();
    auto resource = std::make_unique<int>(0);
    auto build = std::make_unique<PipelineBuild>(
        &workers, [reservation = std::shared_ptr{budget.TryAcquire(true)}, gate, &entered,
                   output = resource.get()] {
            entered.set_value();
            gate.wait();
            *output = 42;
        });
    entered.get_future().wait();
    auto destroy = std::async(std::launch::async, [&] { build.reset(); });
    const bool destruction_held = Pending(destroy);
    release.set_value();
    destroy.get();
    REQUIRE(destruction_held);
    REQUIRE(*resource == 42);
    REQUIRE(budget.Pending() == 0);
}

TEST_CASE("Emission reservations enforce the production fallback and survive budget teardown",
          "[video_core][async_draw][pipeline_build]") {
    auto budget = std::make_unique<EmissionBudget>(2);
    REQUIRE_FALSE(budget->TryAcquire(false));
    auto first = budget->TryAcquire(true);
    auto second = budget->TryAcquire(true);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE_FALSE(budget->TryAcquire(true));
    first.reset();
    REQUIRE(budget->TryAcquire(true));
    REQUIRE(budget->Pending() == 1);
    budget.reset();
    second.reset();
}

TEST_CASE("Cached pipeline prebuild bounds captures and ready pipelines before starting work",
          "[video_core][pipeline_prebuild]") {
    Vulkan::PipelinePrebuild<int, int> cache{2, 100, false};
    auto build = [] { return std::make_unique<int>(42); };
    REQUIRE(cache.Add(1, 60, [&] { return build(); }));
    REQUIRE_FALSE(cache.Add(1, 0, [&] { return build(); }));
    REQUIRE_FALSE(cache.Add(2, 41, [&] { return build(); }));
    REQUIRE(cache.Add(2, 40, [&] { return build(); }));
    REQUIRE_FALSE(cache.Add(3, 0, [&] { return build(); }));
    cache.Stop();
    REQUIRE_FALSE(cache.Add(4, 0, [&] { return build(); }));
    REQUIRE_FALSE(cache.Take(1));
}

TEST_CASE("Cached pipeline prebuild adopts a ready pipeline once without queuing duplicates",
          "[video_core][pipeline_prebuild]") {
    Vulkan::PipelinePrebuild<int, int> cache{2, 100, false};
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    std::atomic<int> builds{};
    REQUIRE(cache.Add(1, 1, [&] {
        ++builds;
        return std::make_unique<int>(42);
    }));
    REQUIRE(cache.Add(2, 1, [&] {
        entered.set_value();
        gate.wait();
        return std::make_unique<int>(7);
    }));
    cache.Start(1);
    entered.get_future().wait();
    auto result = cache.Take(1);
    const bool duplicate = static_cast<bool>(cache.Take(1));
    const bool queued = cache.Add(1, 1, [] { return std::make_unique<int>(99); });
    release.set_value();
    cache.Stop();
    REQUIRE(result);
    REQUIRE(*result == 42);
    REQUIRE_FALSE(duplicate);
    REQUIRE_FALSE(queued);
    REQUIRE(builds == 1);
}

TEST_CASE("Draws claim pending prebuilds without waiting for a worker or its budget",
          "[video_core][pipeline_prebuild]") {
    Vulkan::PipelinePrebuild<int, int> cache{2, 100, false};
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    std::atomic<int> builds{};
    REQUIRE(cache.Add(1, 1, [&] {
        entered.set_value();
        gate.wait();
        ++builds;
        return std::make_unique<int>(42);
    }));
    REQUIRE(cache.Add(2, 1, [&] {
        ++builds;
        return std::make_unique<int>(7);
    }));
    cache.Start(1);
    entered.get_future().wait();
    auto draw = std::async(std::launch::async,
                           [&] { return !cache.Take(1) && !cache.Take(2) && !cache.Take(3); });
    const bool immediate = draw.wait_for(std::chrono::seconds{2}) == std::future_status::ready;
    release.set_value();
    const bool fallback = draw.get();
    cache.Stop();
    REQUIRE(immediate);
    REQUIRE(fallback);
    REQUIRE(builds == 1);
    REQUIRE_FALSE(cache.Take(1));
}

TEST_CASE("Cached pipeline failures release owned inputs and allow normal draw fallback",
          "[video_core][pipeline_prebuild]") {
    for (int failure = 0; failure < 3; ++failure) {
        Vulkan::PipelinePrebuild<int, int> cache{2, 100, false};
        auto input = std::make_shared<int>(42);
        std::weak_ptr<int> weak = input;
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        REQUIRE(cache.Add(1, 1, [input = std::move(input), failure]() -> std::unique_ptr<int> {
            if (failure == 0) {
                throw std::runtime_error("cached translation failed");
            }
            if (failure == 1) {
                throw 42;
            }
            PipelineBuild driver{nullptr,
                                 [] { throw std::runtime_error("driver compile failed"); }};
            return driver.Failed() ? nullptr : std::make_unique<int>(42);
        }));
        REQUIRE(cache.Add(2, 1, [&] {
            entered.set_value();
            gate.wait();
            return std::make_unique<int>(7);
        }));
        cache.Start(1);
        entered.get_future().wait();
        const bool released = weak.expired();
        const bool fallback = !cache.Take(1);
        release.set_value();
        cache.Stop();
        REQUIRE(released);
        REQUIRE(fallback);
    }
}

TEST_CASE("Prebuild cancellation joins active work before destroying external resources",
          "[video_core][pipeline_prebuild]") {
    auto resource = std::make_unique<int>(0);
    auto cache = std::make_unique<Vulkan::PipelinePrebuild<int, int>>(2, 100, false);
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    REQUIRE(cache->Add(1, 1, [&, output = resource.get()] {
        entered.set_value();
        gate.wait();
        *output = 42;
        return std::make_unique<int>(7);
    }));
    auto queued_input = std::make_shared<int>(42);
    std::weak_ptr<int> weak = queued_input;
    REQUIRE(cache->Add(
        2, 1, [input = std::move(queued_input)] { return std::make_unique<int>(*input); }));
    cache->Start(1);
    entered.get_future().wait();
    auto destroy = std::async(std::launch::async, [&] { cache.reset(); });
    const bool joined = Pending(destroy);
    release.set_value();
    destroy.get();
    REQUIRE(joined);
    REQUIRE(*resource == 42);
    REQUIRE(weak.expired());
}

TEST_CASE("Broken parallel compilation policy prevents all speculative work",
          "[video_core][pipeline_prebuild]") {
    Vulkan::PipelinePrebuild<int, int> cache{64, 16 * 1024 * 1024, true};
    auto input = std::make_shared<int>(42);
    std::weak_ptr<int> weak = input;
    bool built = false;
    REQUIRE_FALSE(cache.Add(1, 1, [&, input = std::move(input)] {
        built = true;
        return std::make_unique<int>(*input);
    }));
    cache.Start(2);
    cache.Stop();
    REQUIRE_FALSE(built);
    REQUIRE(weak.expired());
}

TEST_CASE("Cancelling startup prebuilds releases captures without running them",
          "[video_core][pipeline_prebuild]") {
    Vulkan::PipelinePrebuild<int, int> cache{2, 100, false};
    auto input = std::make_shared<int>(42);
    std::weak_ptr<int> weak = input;
    bool built = false;
    REQUIRE(cache.Add(1, 1, [&, input = std::move(input)] {
        built = true;
        return std::make_unique<int>(*input);
    }));
    cache.Stop();
    cache.Start(2);
    REQUIRE_FALSE(built);
    REQUIRE(weak.expired());
}

TEST_CASE("Concurrent prebuild completion and adoption never return a pipeline twice",
          "[video_core][pipeline_prebuild]") {
    for (int iteration = 0; iteration < 32; ++iteration) {
        Vulkan::PipelinePrebuild<int, int> cache{2, 100, false};
        REQUIRE(cache.Add(1, 1, [] { return std::make_unique<int>(42); }));
        REQUIRE(cache.Add(2, 1, [] { return std::make_unique<int>(7); }));
        cache.Start(2);
        auto first = std::async(std::launch::async, [&] { return cache.Take(1); });
        auto second = std::async(std::launch::async, [&] { return cache.Take(1); });
        const auto a = first.get();
        const auto b = second.get();
        cache.Stop();
        REQUIRE_FALSE((a && b));
        if (a) {
            REQUIRE(*a == 42);
        }
        if (b) {
            REQUIRE(*b == 42);
        }
    }
}
