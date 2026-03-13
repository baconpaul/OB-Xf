/*
 * OB-Xf - a continuation of the last open source version of OB-Xd.
 *
 * OB-Xd was originally written by Vadim Filatov, and then a version
 * was released under the GPL3 at https://github.com/reales/OB-Xd.
 * Subsequently, the product was continued by DiscoDSP and the copyright
 * holders as an excellent closed source product. For more info,
 * see "HISTORY.md" in the root of this repository.
 *
 * This repository is a successor to the last open source release,
 * a version marked as 2.11. Copyright 2013-2025 by the authors
 * as indicated in the original release, and subsequent authors
 * per the GitHub transaction log.
 *
 * OB-Xf is released under the GNU General Public Licence v3 or later
 * (GPL-3.0-or-later). The license is found in the file "LICENSE"
 * in the root of this repository or at:
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 * Source code is available at https://github.com/surge-synthesizer/OB-Xf
 */

#ifndef OBXF_SRC_ENGINE_VOICEMATRIX_H
#define OBXF_SRC_ENGINE_VOICEMATRIX_H

#include <array>
#include <string>
#include <unordered_set>

#include <juce_core/juce_core.h>

#include "configuration.h"
#include "SynthParam.h"

/*
 * VoiceMatrix: per-synth modulation routing from MPE/voice sources to per-voice targets.
 * Sources are enumerated below; targets are SynthParam ID strings.
 * The matrix has numMatrixRows rows, each with a source, target, and depth.
 */

// ---------------------------------------------------------------------------
// Source enum — use string conversion for stable streaming (not int values)
// ---------------------------------------------------------------------------
enum class MatrixSource
{
    VoiceBend,
    ChannelPressure,
    Timbre,
    Velocity,
    ReleaseVelocity,
    None
};

inline std::string matrixSourceToString(MatrixSource src)
{
    switch (src)
    {
    case MatrixSource::VoiceBend:
        return "VoiceBend";
    case MatrixSource::ChannelPressure:
        return "ChannelPressure";
    case MatrixSource::Timbre:
        return "Timbre";
    case MatrixSource::Velocity:
        return "Velocity";
    case MatrixSource::ReleaseVelocity:
        return "ReleaseVelocity";
    case MatrixSource::None:
    default:
        return "None";
    }
}

inline MatrixSource matrixSourceFromString(const std::string &s)
{
    if (s == "VoiceBend")
        return MatrixSource::VoiceBend;
    if (s == "ChannelPressure")
        return MatrixSource::ChannelPressure;
    if (s == "Timbre")
        return MatrixSource::Timbre;
    if (s == "Velocity")
        return MatrixSource::Velocity;
    if (s == "ReleaseVelocity")
        return MatrixSource::ReleaseVelocity;
    return MatrixSource::None;
}

/*
 * VoiceMatrixAdjustments: per-voice modulation accumulator.
 * Contains one float per valid matrix target — keep this in sync with
 * isValidMatrixTarget() below.
 */
struct VoiceMatrixAdjustments
{
    // Keep in sync with isValidMatrixTarget()
    float filterCutoff{0.f};
    float filterResonance{0.f};
    float osc1Pitch{0.f};
    float osc2Pitch{0.f};
    float osc2Detune{0.f};
    float osc2PWOffset{0.f};
    float osc1Vol{0.f};
    float osc2Vol{0.f};
    float noiseVol{0.f};
    float ringModVol{0.f};
    float noiseColor{0.f};

    void clear()
    {
        filterCutoff = 0.f;
        filterResonance = 0.f;
        osc1Pitch = 0.f;
        osc2Pitch = 0.f;
        osc2Detune = 0.f;
        osc2PWOffset = 0.f;
        osc1Vol = 0.f;
        osc2Vol = 0.f;
        noiseVol = 0.f;
        ringModVol = 0.f;
        noiseColor = 0.f;
    }
};

/*
 * VoiceMatrixRanges: natural scale for each modulation target.
 * +/-1 source value maps to +/- the range value in the target's native units.
 * Keep in sync with VoiceMatrixAdjustments above and isValidMatrixTarget() below.
 */
struct VoiceMatrixRanges
{
    // Pitch targets: +/-1 = +/-48 semitones
    static constexpr float filterCutoff{48.f};
    static constexpr float osc1Pitch{48.f};
    static constexpr float osc2Pitch{48.f};
    // All other targets: +/-1 = +/-1 (full range)
    static constexpr float filterResonance{1.f};
    static constexpr float osc2Detune{1.f};
    static constexpr float osc2PWOffset{1.f};
    static constexpr float osc1Vol{1.f};
    static constexpr float osc2Vol{1.f};
    static constexpr float noiseVol{1.f};
    static constexpr float ringModVol{1.f};
    static constexpr float noiseColor{1.f};
};

// ---------------------------------------------------------------------------
// Valid modulation targets — keep in sync with VoiceMatrixAdjustments above
// ---------------------------------------------------------------------------
inline bool isValidMatrixTarget(const std::string &tgt)
{
    static const std::unordered_set<std::string> validTargets = {
        SynthParam::ID::FilterCutoff,
        SynthParam::ID::FilterResonance,
        SynthParam::ID::Osc1Pitch,
        SynthParam::ID::Osc2Pitch,
        SynthParam::ID::Osc2Detune,
        SynthParam::ID::Osc2PWOffset,
        SynthParam::ID::Osc1Vol,
        SynthParam::ID::Osc2Vol,
        SynthParam::ID::NoiseVol,
        SynthParam::ID::RingModVol,
        SynthParam::ID::NoiseColor,
    };
    return validTargets.count(tgt) > 0;
}

// ---------------------------------------------------------------------------
// A single matrix row
// ---------------------------------------------------------------------------
struct MatrixRow
{
    MatrixSource source{MatrixSource::None};
    std::string target{}; // SynthParam::ID string, empty = unassigned
    float depth{0.f};     // -1..1

    bool isActive() const { return source != MatrixSource::None && !target.empty(); }
};

// ---------------------------------------------------------------------------
// The matrix itself — lives on the synth, streams in the patch XML
// ---------------------------------------------------------------------------
struct VoiceMatrix
{
    std::array<MatrixRow, numMatrixRows> rows{};

    /* Returns false if source/target are invalid or idx is out of range */
    bool setModulation(const std::string &src, const std::string &tgt, float depth, int idx)
    {
        if (idx < 0 || idx >= numMatrixRows)
            return false;

        auto s = matrixSourceFromString(src);
        if (s == MatrixSource::None)
            return false;

        if (!isValidMatrixTarget(tgt))
            return false;

        rows[idx].source = s;
        rows[idx].target = tgt;
        rows[idx].depth = depth;
        return true;
    }

    void clearRow(int idx)
    {
        if (idx >= 0 && idx < numMatrixRows)
            rows[idx] = MatrixRow{};
    }

    void clear() { rows.fill(MatrixRow{}); }

    // -----------------------------------------------------------------------
    // XML streaming — call toElement / fromElement from patch save/load
    // -----------------------------------------------------------------------
    std::unique_ptr<juce::XmlElement> toElement() const
    {
        auto el = std::make_unique<juce::XmlElement>("VoiceMatrix");
        for (int i = 0; i < numMatrixRows; ++i)
        {
            const auto &row = rows[i];
            if (!row.isActive())
                continue;

            auto *rowEl = new juce::XmlElement("row");
            rowEl->setAttribute("idx", i);
            rowEl->setAttribute("source", matrixSourceToString(row.source));
            rowEl->setAttribute("target", row.target);
            rowEl->setAttribute("depth", row.depth);
            el->addChildElement(rowEl);
        }
        return el;
    }

    void fromElement(const juce::XmlElement *el)
    {
        clear();
        if (!el)
            return;

        for (auto *rowEl : el->getChildIterator())
        {
            int idx = rowEl->getIntAttribute("idx", -1);
            if (idx < 0 || idx >= numMatrixRows)
                continue;

            rows[idx].source =
                matrixSourceFromString(rowEl->getStringAttribute("source").toStdString());
            rows[idx].target = rowEl->getStringAttribute("target").toStdString();
            rows[idx].depth = static_cast<float>(rowEl->getDoubleAttribute("depth", 0.0));
        }
    }
};

// ---------------------------------------------------------------------------
// Matrix presets — THROWAWAY / for testing only
// ---------------------------------------------------------------------------
struct MatrixPreset
{
    std::string name;
    std::array<MatrixRow, numMatrixRows> rows{};
};

inline std::vector<MatrixPreset> getMatrixPresets()
{
    std::vector<MatrixPreset> presets;

    // Preset 1: Timbre → Cutoff
    {
        MatrixPreset p;
        p.name = "Timbre to Cutoff";
        p.rows[0] = {MatrixSource::Timbre, SynthParam::ID::FilterCutoff, 0.7f};
        presets.push_back(p);
    }

    // Preset 2: Pressure → Pitch, Timbre → Cutoff
    {
        MatrixPreset p;
        p.name = "Pressure to Pitch + Timbre to Cutoff";
        p.rows[0] = {MatrixSource::ChannelPressure, SynthParam::ID::Osc1Pitch, 0.3f};
        p.rows[1] = {MatrixSource::ChannelPressure, SynthParam::ID::Osc2Pitch, 0.3f};
        p.rows[2] = {MatrixSource::Timbre, SynthParam::ID::FilterCutoff, 0.7f};
        presets.push_back(p);
    }

    // Preset 3: Velocity → Cutoff + Resonance
    {
        MatrixPreset p;
        p.name = "Velocity to Filter";
        p.rows[0] = {MatrixSource::Velocity, SynthParam::ID::FilterCutoff, 0.5f};
        p.rows[1] = {MatrixSource::Velocity, SynthParam::ID::FilterResonance, 0.3f};
        presets.push_back(p);
    }

    // Preset 4: VoiceBend → Cutoff (expressive filter sweep)
    {
        MatrixPreset p;
        p.name = "Bend to Cutoff";
        p.rows[0] = {MatrixSource::VoiceBend, SynthParam::ID::FilterCutoff, 0.5f};
        presets.push_back(p);
    }

    // Preset 5: Timbre → Osc Mix + Cutoff
    {
        MatrixPreset p;
        p.name = "Timbre to Mix + Cutoff";
        p.rows[0] = {MatrixSource::Timbre, SynthParam::ID::Osc1Vol, 0.6f};
        p.rows[1] = {MatrixSource::Timbre, SynthParam::ID::FilterCutoff, 0.4f};
        presets.push_back(p);
    }

    return presets;
}

#endif // OBXF_SRC_ENGINE_VOICEMATRIX_H
