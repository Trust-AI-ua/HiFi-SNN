// SANA-FE soma plugin for track A3 (results/preregistration/2026-11-17_neuro_stream.md, amendment 2).
//
// SANA-FE's built-in Loihi LIF (src/models.cpp, LoihiLifModel, v2.2.9) with one change: the
// subtractive reset is applied one step after the spike, as in snnTorch's Leaky
// (reset_mechanism="subtract") and in the Lava process src/lava_sublif:
//     v_t = Q(leak * v_{t-1}) + bias + I_t - threshold * s_{t-1},  s_t = [v_t > threshold]
// where Q truncates toward zero on the 1/64 grid (Loihi's 6-bit MSB alignment), exactly as the
// built-in model does. No reset in the step of the spike. Status (idle / updated / fired) is
// reported as the built-in model reports it, so the architecture's energy and latency costs of
// the soma unit apply unchanged. Attributes: threshold, bias, leak_decay, force_update.
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"
#include "print.hpp"

class LoihiLifDelayed : public sanafe::SomaUnit
{
public:
    struct Cx
    {
        double v{0.0};
        double bias{0.0};
        double leak{1.0};
        double threshold{0.0};
        bool force{false};
        bool prev_fired{false};
        long int steps{0L};
    };
    std::vector<Cx> cx;

    LoihiLifDelayed() { register_attributes({"threshold", "bias", "leak_decay", "force_update"}); }

    Cx &at(const size_t a)
    {
        if (a >= cx.size()) { cx.resize(a + 1); }
        return cx[a];
    }

    double get_potential(const size_t a) override { return at(a).v; }

    void reset() override
    {
        for (Cx &c : cx) { c.v = 0.0; c.prev_fired = false; }
    }

    void set_attribute_hw(const std::string & /*name*/, const sanafe::ModelAttribute & /*p*/) override {}

    void set_attribute_neuron(const size_t a, const std::string &name, const sanafe::ModelAttribute &p) override
    {
        Cx &c = at(a);
        if (name == "threshold") { c.threshold = static_cast<double>(p); }
        else if (name == "bias") { c.bias = static_cast<double>(p); }
        else if (name == "leak_decay") { c.leak = static_cast<double>(p); }
        else if (name == "force_update") { c.force = static_cast<bool>(p); }
    }

    sanafe::PipelineResult update(const size_t a, const std::optional<double> current_in,
            const long int /*timestep*/) override
    {
        Cx &c = at(a);
        sanafe::NeuronStatus status = sanafe::idle;
        if ((std::fabs(c.v) > 0.0) || current_in.has_value() || (std::fabs(c.bias) > 0.0) || c.force)
        {
            status = sanafe::updated;
        }
        if (c.steps > 0) { c.v *= c.leak; }
        c.v = static_cast<int>(c.v * 64.0) / 64.0;                 // as LoihiLifModel::loihi_quantize
        c.v += c.bias + current_in.value_or(0.0) - (c.prev_fired ? c.threshold : 0.0);
        const bool fired = c.v > c.threshold;
        c.prev_fired = fired;
        if (fired) { status = sanafe::fired; }
        ++c.steps;
        return {std::nullopt, status, std::nullopt, std::nullopt};
    }
};

extern "C" sanafe::PipelineUnit *create_loihi_lif_delayed()
{
    return (sanafe::PipelineUnit *) new LoihiLifDelayed();
}
