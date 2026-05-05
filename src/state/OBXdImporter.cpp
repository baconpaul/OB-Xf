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

#include "OBXdImporter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <juce_audio_processors/juce_audio_processors.h>

#include "Constants.h"
#include "ParameterList.h"
#include "SynthParam.h"
#include "configuration.h"

namespace
{

// OB-Xd ObxdParameters indices, frozen as they are written to disk.
// Mirrors third_party/OB-Xd/Source/Engine/ParamsEnum.h tag 2.17.
enum XdParam : int
{
    XD_UNDEFINED = 0,
    XD_MIDILEARN = 1,
    XD_VOLUME = 2,
    XD_VOICE_COUNT = 3,
    XD_TUNE = 4,
    XD_OCTAVE = 5,
    XD_BENDRANGE = 6,
    XD_BENDOSC2 = 7,
    XD_LEGATOMODE = 8,
    XD_BENDLFORATE = 9,
    XD_VFLTENV = 10,
    XD_VAMPENV = 11,
    XD_ASPLAYEDALLOCATION = 12,
    XD_PORTAMENTO = 13,
    XD_UNISON = 14,
    XD_UDET = 15,
    XD_OSC2_DET = 16,
    XD_LFOFREQ = 17,
    XD_LFOSINWAVE = 18,
    XD_LFOSQUAREWAVE = 19,
    XD_LFOSHWAVE = 20,
    XD_LFO1AMT = 21,
    XD_LFO2AMT = 22,
    XD_LFOOSC1 = 23,
    XD_LFOOSC2 = 24,
    XD_LFOFILTER = 25,
    XD_LFOPW1 = 26,
    XD_LFOPW2 = 27,
    XD_OSC2HS = 28,
    XD_XMOD = 29,
    XD_OSC1P = 30,
    XD_OSC2P = 31,
    XD_OSCQuantize = 32,
    XD_OSC1Saw = 33,
    XD_OSC1Pul = 34,
    XD_OSC2Saw = 35,
    XD_OSC2Pul = 36,
    XD_PW = 37,
    XD_BRIGHTNESS = 38,
    XD_ENVPITCH = 39,
    XD_OSC1MIX = 40,
    XD_OSC2MIX = 41,
    XD_NOISEMIX = 42,
    XD_FLT_KF = 43,
    XD_CUTOFF = 44,
    XD_RESONANCE = 45,
    XD_MULTIMODE = 46,
    XD_FILTER_WARM = 47,
    XD_BANDPASS = 48,
    XD_FOURPOLE = 49,
    XD_ENVELOPE_AMT = 50,
    XD_LATK = 51,
    XD_LDEC = 52,
    XD_LSUS = 53,
    XD_LREL = 54,
    XD_FATK = 55,
    XD_FDEC = 56,
    XD_FSUS = 57,
    XD_FREL = 58,
    XD_ENVDER = 59,
    XD_FILTERDER = 60,
    XD_PORTADER = 61,
    XD_PAN1 = 62,
    XD_PAN2 = 63,
    XD_PAN3 = 64,
    XD_PAN4 = 65,
    XD_PAN5 = 66,
    XD_PAN6 = 67,
    XD_PAN7 = 68,
    XD_PAN8 = 69,
    XD_UNLEARN = 70,
    XD_ECONOMY_MODE = 71,
    XD_LFO_SYNC = 72,
    XD_PW_ENV = 73,
    XD_PW_ENV_BOTH = 74,
    XD_ENV_PITCH_BOTH = 75,
    XD_FENV_INVERT = 76,
    XD_PW_OSC2_OFS = 77,
    XD_LEVEL_DIF = 78,
    XD_SELF_OSC_PUSH = 79,
    XD_PARAM_COUNT = 80,
};

// Re-implementations of OB-Xd's logsc/linsc plus their inverses, used for
// rescaling normalized values across the two engines. Matches the OB-Xd
// definitions in Source/Engine/AudioUtils.h.
inline float xdLogsc(float p, float lo, float hi, float rolloff = 19.f)
{
    return ((std::exp(p * std::log(rolloff + 1.f)) - 1.f) / rolloff) * (hi - lo) + lo;
}

inline float xdInvLinsc(float y, float lo, float hi)
{
    if (hi == lo)
        return 0.f;
    return juce::jlimit(0.f, 1.f, (y - lo) / (hi - lo));
}

inline float xdInvLogsc(float y, float lo, float hi, float rolloff = 19.f)
{
    if (hi == lo)
        return 0.f;
    const float t = rolloff * (y - lo) / (hi - lo) + 1.f;
    if (t <= 0.f)
        return 0.f;
    return juce::jlimit(0.f, 1.f, std::log(t) / std::log(rolloff + 1.f));
}

// Read attribute Val_<k>, falling back to bare <k> for the legacy schema.
float readXdAttribute(const juce::XmlElement &e, int k, float fallback = 0.f)
{
    const auto vKey = "Val_" + juce::String(k);
    if (e.hasAttribute(vKey))
        return static_cast<float>(e.getDoubleAttribute(vKey, fallback));
    const auto bareKey = juce::String(k);
    if (e.hasAttribute(bareKey))
        return static_cast<float>(e.getDoubleAttribute(bareKey, fallback));
    return fallback;
}

// Map OB-Xd LFO1 sync rate (9 buckets) to OB-Xf's 21-bucket synced table.
// Lookup via (int)(v*8) -> XF normalized rate. See LFOSync.md.
float mapLfoSyncedRate(float vXd)
{
    static constexpr int xdToXf[9] = {1, 4, 5, 7, 10, 11, 13, 15, 16};
    const int kXd = juce::jlimit(0, 8, static_cast<int>(vXd * 8.f));
    return static_cast<float>(xdToXf[kXd]) / static_cast<float>(syncedRatesCount - 1);
}

void seedDefaults(Program &program)
{
    program.values.clear();
    for (const auto &param : ParameterList)
    {
        program.values[param.ID] = param.meta.naturalToNormalized01(param.meta.defaultVal);
    }
}

void translateLfoFreq(float vXd, bool synced, Program &program)
{
    using namespace SynthParam;

    if (synced)
    {
        program.values[ID::LFO1Rate] = mapLfoSyncedRate(vXd);
    }
    else
    {
        const float hzXd = xdLogsc(vXd, 0.f, 50.f, 120.f);
        program.values[ID::LFO1Rate] = xdInvLogsc(hzXd, 0.f, 250.f, 3775.f);
    }
}

void translateAttackTime(float vXd, const juce::String &id, float lo, float hi, bool compensate,
                         Program &program)
{
    if (!compensate)
    {
        program.values[id] = vXd;
        return;
    }

    const float msXd = xdLogsc(vXd, lo, hi, 900.f);
    const float msXf = msXd / 3.f;
    program.values[id] = xdInvLogsc(msXf, lo, hi, 900.f);
}

float lfoBoolToBlend(float v) { return v >= 0.5f ? 0.f : 0.5f; }

float lfoBoolToTriState(float v)
{
    // OB-Xf tri-state {Off, On, Inverted} stored as int 0..2 normalized.
    // 0 -> 0/2 = 0; 1 -> 1/2 = 0.5; 2 -> 1.0.
    return v >= 0.5f ? 0.5f : 0.f;
}

void writeOscMix(float vXd, const juce::String &id, Program &program) { program.values[id] = vXd; }

} // namespace

void OBXdImporter::translateProgramFromXml(const juce::XmlElement &e, Program &program,
                                           std::vector<std::string> &warnings, const Options &opt)
{
    using namespace SynthParam;

    seedDefaults(program);

    // Override defaults that the plan calls out as not matching OB-Xd hard-wired behavior.
    program.values[ID::Osc2Keytrack] = 1.f;
    program.values[ID::EnvToPitchInvert] = 0.f;
    program.values[ID::EnvToPWInvert] = 0.f;
    program.values[ID::RingModVol] = 0.f;
    program.values[ID::NoiseColor] = 0.f;
    program.values[ID::Filter4PoleXpander] = 0.f;
    program.values[ID::FilterXpanderMode] = 0.f;
    program.values[ID::AmpEnvAttackCurve] = 0.f;
    program.values[ID::FilterEnvAttackCurve] = 0.f;
    program.values[ID::LFO1PW] = 0.f;
    program.values[ID::LFO1ToVolume] = 0.f;
    program.values[ID::VibratoWave] = 0.f;
    program.values[ID::NotePriority] = 0.f;
    program.values[ID::EnvLegatoMode] = 0.f;

    const bool newFormat = e.hasAttribute("voiceCount");

    // Pre-fetch values we need to consult for cross-parameter decisions.
    const bool lfoSync = readXdAttribute(e, XD_LFO_SYNC) > 0.5f;

    auto val = [&](int k) { return readXdAttribute(e, k); };

    // ---- Master / Global ----

    program.values[ID::Volume] = val(XD_VOLUME);
    program.values[ID::Tune] = val(XD_TUNE);
    program.values[ID::Transpose] = val(XD_OCTAVE);

    {
        float vc = val(XD_VOICE_COUNT);
        if (!newFormat)
            vc *= 0.25f;

        // OB-Xd: voices = 1 + round(vc*31). OB-Xf engine: 1 + (int)(v*32)
        // produces 33 at v=1, so write the OB-Xd integer back in OB-Xf's
        // (n-1)/32 normalized space, capped to keep us inside MAX_VOICES.
        const int xdVoices =
            juce::jlimit(1, MAX_VOICES, 1 + static_cast<int>(std::round(vc * 31.f)));
        program.values[ID::Polyphony] =
            static_cast<float>(xdVoices - 1) / static_cast<float>(MAX_VOICES);
    }

    {
        const float br = val(XD_BENDRANGE);
        const int range = (br > 0.5f) ? 12 : 2;
        const float n = static_cast<float>(range) / static_cast<float>(MAX_BEND_RANGE);
        program.values[ID::BendUpRange] = n;
        program.values[ID::BendDownRange] = n;
    }
    program.values[ID::BendOsc2Only] = val(XD_BENDOSC2);

    {
        // OB-Xd vibrato: setFrequency(logsc(v, 3, 10)) Hz
        // OB-Xf vibrato: setRate(linsc(v, 2, 12)) Hz
        const float hzXd = xdLogsc(val(XD_BENDLFORATE), 3.f, 10.f);
        program.values[ID::VibratoRate] = xdInvLinsc(hzXd, 2.f, 12.f);
    }

    program.values[ID::Portamento] = val(XD_PORTAMENTO);
    program.values[ID::Unison] = val(XD_UNISON);

    {
        // OB-Xd UDET: logsc(v, 0.001, 0.90); OB-Xf UnisonDetune: logsc(v, 0.001, 1.0)
        const float dXd = xdLogsc(val(XD_UDET), 0.001f, 0.90f);
        program.values[ID::UnisonDetune] = xdInvLogsc(dXd, 0.001f, 1.0f);
    }

    // OB-Xd legato has different semantics than OB-Xf's enum — default Both.
    if (val(XD_LEGATOMODE) > 0.f)
    {
        warnings.emplace_back(
            "LEGATOMODE: OB-Xd flag combinations are not mapped; defaulted to 'Both'.");
    }

    if (val(XD_ASPLAYEDALLOCATION) > 0.f)
    {
        warnings.emplace_back(
            "ASPLAYEDALLOCATION: not mapped; OB-Xf NotePriority defaulted to 'Last'.");
    }

    if (val(XD_ECONOMY_MODE) > 0.f)
        warnings.emplace_back("ECONOMY_MODE: dropped (no equivalent in OB-Xf).");
    if (val(XD_OSCQuantize) > 0.f)
        warnings.emplace_back("OSCQuantize: dropped (no pitch-quantize knob in OB-Xf).");

    // ---- Oscillators ----

    program.values[ID::Osc1Pitch] = val(XD_OSC1P);
    program.values[ID::Osc2Pitch] = val(XD_OSC2P);
    program.values[ID::Osc2Detune] = val(XD_OSC2_DET);
    program.values[ID::Osc1SawWave] = val(XD_OSC1Saw);
    program.values[ID::Osc1PulseWave] = val(XD_OSC1Pul);
    program.values[ID::Osc2SawWave] = val(XD_OSC2Saw);
    program.values[ID::Osc2PulseWave] = val(XD_OSC2Pul);
    program.values[ID::OscSync] = val(XD_OSC2HS);

    // OB-Xd XMOD: v*24 semis. OB-Xf OscCrossmod: v*48.
    program.values[ID::OscCrossmod] = val(XD_XMOD) * 0.5f;

    program.values[ID::OscPW] = val(XD_PW);

    // OB-Xd PW_OSC2_OFS: linsc(v, 0, 0.75). OB-Xf Osc2PWOffset: linsc(v, 0, 0.95).
    program.values[ID::Osc2PWOffset] = val(XD_PW_OSC2_OFS) * (0.75f / 0.95f);

    // OB-Xd PW_ENV: linsc(v, 0, 0.85); OB-Xf EnvToPWAmount: linsc(v, 0, 1.05555…).
    program.values[ID::EnvToPWAmount] = val(XD_PW_ENV) * (0.85f / 1.0555555555f);

    program.values[ID::EnvToPWBothOscs] = val(XD_PW_ENV_BOTH);

    // OB-Xd ENVPITCH: v*36 semitones; OB-Xf EnvToPitchAmount: v*40 (sustain comp).
    program.values[ID::EnvToPitchAmount] = val(XD_ENVPITCH) * (36.f / 40.f);

    program.values[ID::EnvToPitchBothOscs] = val(XD_ENV_PITCH_BOTH);
    program.values[ID::OscBrightness] = val(XD_BRIGHTNESS);

    // ---- Mixer ----

    writeOscMix(val(XD_OSC1MIX), ID::Osc1Vol, program);
    writeOscMix(val(XD_OSC2MIX), ID::Osc2Vol, program);
    // OB-Xd noise mix runs the value through logsc(v, 0, 1, 35) before the engine.
    program.values[ID::NoiseVol] = xdLogsc(val(XD_NOISEMIX), 0.f, 1.f, 35.f);

    // ---- Filter ----

    program.values[ID::FilterCutoff] = val(XD_CUTOFF);
    program.values[ID::FilterResonance] = val(XD_RESONANCE);
    program.values[ID::FilterMode] = val(XD_MULTIMODE);
    program.values[ID::Filter2PoleBPBlend] = val(XD_BANDPASS);
    program.values[ID::Filter4PoleMode] = val(XD_FOURPOLE);
    program.values[ID::HQMode] = val(XD_FILTER_WARM);
    program.values[ID::Filter2PolePush] = val(XD_SELF_OSC_PUSH);
    program.values[ID::FilterKeyTrack] = val(XD_FLT_KF);
    program.values[ID::FilterEnvAmount] = val(XD_ENVELOPE_AMT);
    program.values[ID::FilterEnvInvert] = val(XD_FENV_INVERT);

    // ---- Envelopes ----

    translateAttackTime(val(XD_LATK), ID::AmpEnvAttack, 4.f, 60000.f, opt.envAttackCompat, program);
    program.values[ID::AmpEnvDecay] = val(XD_LDEC);
    program.values[ID::AmpEnvSustain] = val(XD_LSUS);
    program.values[ID::AmpEnvRelease] = val(XD_LREL);

    translateAttackTime(val(XD_FATK), ID::FilterEnvAttack, 1.f, 60000.f, opt.envAttackCompat,
                        program);
    program.values[ID::FilterEnvDecay] = val(XD_FDEC);
    program.values[ID::FilterEnvSustain] = val(XD_FSUS);
    program.values[ID::FilterEnvRelease] = val(XD_FREL);

    program.values[ID::VelToAmpEnv] = val(XD_VAMPENV);
    program.values[ID::VelToFilterEnv] = val(XD_VFLTENV);

    // ---- LFO 1 ----

    program.values[ID::LFO1TempoSync] = lfoSync ? 1.f : 0.f;
    translateLfoFreq(val(XD_LFOFREQ), lfoSync, program);
    if (lfoSync)
    {
        warnings.emplace_back("LFO_SYNC: OB-Xd's sync was unreliable; OB-Xf will sync correctly so "
                              "the patch may sound different. Selected rate bucket is preserved.");
    }

    program.values[ID::LFO1Wave1] = lfoBoolToBlend(val(XD_LFOSINWAVE));
    program.values[ID::LFO1Wave2] = lfoBoolToBlend(val(XD_LFOSQUAREWAVE));
    program.values[ID::LFO1Wave3] = lfoBoolToBlend(val(XD_LFOSHWAVE));

    program.values[ID::LFO1ModAmount1] = val(XD_LFO1AMT);
    program.values[ID::LFO1ModAmount2] = val(XD_LFO2AMT);

    program.values[ID::LFO1ToOsc1Pitch] = lfoBoolToTriState(val(XD_LFOOSC1));
    program.values[ID::LFO1ToOsc2Pitch] = lfoBoolToTriState(val(XD_LFOOSC2));
    program.values[ID::LFO1ToFilterCutoff] = lfoBoolToTriState(val(XD_LFOFILTER));
    program.values[ID::LFO1ToOsc1PW] = lfoBoolToTriState(val(XD_LFOPW1));
    program.values[ID::LFO1ToOsc2PW] = lfoBoolToTriState(val(XD_LFOPW2));

    // ---- Voice variation / pan ----

    program.values[ID::PortamentoSlop] = val(XD_PORTADER);
    program.values[ID::FilterSlop] = val(XD_FILTERDER);
    program.values[ID::EnvelopeSlop] = val(XD_ENVDER);
    program.values[ID::LevelSlop] = val(XD_LEVEL_DIF);

    program.values[ID::PanVoice1] = val(XD_PAN1);
    program.values[ID::PanVoice2] = val(XD_PAN2);
    program.values[ID::PanVoice3] = val(XD_PAN3);
    program.values[ID::PanVoice4] = val(XD_PAN4);
    program.values[ID::PanVoice5] = val(XD_PAN5);
    program.values[ID::PanVoice6] = val(XD_PAN6);
    program.values[ID::PanVoice7] = val(XD_PAN7);
    program.values[ID::PanVoice8] = val(XD_PAN8);

    // ---- Metadata ----

    program.setName(e.getStringAttribute("programName", "Init"));
    program.setAuthor("");
    program.setLicense("");
    program.setCategory("None");
    program.setProject("");
}

namespace
{

// Strict header check: 'CcnK' and fxID == 'Obxd'.
const fxChunkSet *asObxdChunkSet(const void *data, size_t size)
{
    if (size < sizeof(fxChunkSet) - 8)
        return nullptr;

    const auto *set = static_cast<const fxChunkSet *>(data);

    if (!compareMagic(set->chunkMagic, "CcnK"))
        return nullptr;
    if (!compareMagic(set->fxID, "Obxd"))
        return nullptr;
    if (fxbSwap(set->version) > fxbVersionNum)
        return nullptr;
    return set;
}

// Single program (FPCh) chunk extraction.
bool readFPChChunk(const void *data, size_t size, const void *&outChunkData, int &outChunkSize)
{
    const auto *set = static_cast<const fxProgramSet *>(data);
    const size_t chunkSize = static_cast<size_t>(fxbSwap(set->chunkSize));
    if (chunkSize + sizeof(fxProgramSet) - 8 > size)
        return false;
    outChunkData = set->chunk;
    outChunkSize = static_cast<int>(chunkSize);
    return true;
}

// Bank (FBCh) chunk extraction.
bool readFBChChunk(const void *data, size_t size, const void *&outChunkData, int &outChunkSize)
{
    const auto *set = static_cast<const fxChunkSet *>(data);
    const size_t chunkSize = static_cast<size_t>(fxbSwap(set->chunkSize));
    if (chunkSize + sizeof(fxChunkSet) - 8 > size)
        return false;
    outChunkData = set->chunk;
    outChunkSize = static_cast<int>(chunkSize);
    return true;
}

} // namespace

bool OBXdImporter::isOBXdData(const void *data, size_t size)
{
    return asObxdChunkSet(data, size) != nullptr;
}

bool OBXdImporter::visitPrograms(
    const void *data, size_t size,
    const std::function<void(int, bool, const juce::XmlElement &)> &visit)
{
    const auto *set = asObxdChunkSet(data, size);
    if (!set)
        return false;

    const void *chunk = nullptr;
    int chunkSize = 0;

    const bool isFPCh = compareMagic(set->fxMagic, "FPCh");
    const bool isFBCh = compareMagic(set->fxMagic, "FBCh");

    if (isFPCh)
    {
        if (!readFPChChunk(data, size, chunk, chunkSize))
            return false;
    }
    else if (isFBCh)
    {
        if (!readFBChChunk(data, size, chunk, chunkSize))
            return false;
    }
    else
    {
        return false;
    }

    auto xml = juce::AudioProcessor::getXmlFromBinary(chunk, chunkSize);
    if (!xml || !xml->hasTagName("discoDSP"))
        return false;

    if (isFPCh)
    {
        visit(0, true, *xml);
        return true;
    }

    const int currentProgram = xml->getIntAttribute("currentProgram", 0);
    const auto *programsNode = xml->getChildByName("programs");
    if (!programsNode)
        return false;

    int idx = 0;
    bool sawAny = false;
    for (auto *programEl : programsNode->getChildIterator())
    {
        if (!programEl->hasTagName("program"))
            continue;
        visit(idx, idx == currentProgram, *programEl);
        sawAny = true;
        idx++;
        if (idx >= 128)
            break;
    }

    return sawAny;
}

bool OBXdImporter::importSingleOnto(const void *data, size_t size, Program &outProgram,
                                    std::vector<std::string> *outWarnings, const Options &opt)
{
    bool wrote = false;
    std::vector<std::string> warnings;

    visitPrograms(data, size, [&](int /*idx*/, bool isCurrent, const juce::XmlElement &programEl) {
        if (!isCurrent || wrote)
            return;
        translateProgramFromXml(programEl, outProgram, warnings, opt);
        wrote = true;
    });

    if (outWarnings)
    {
        for (auto &w : warnings)
            outWarnings->emplace_back(std::move(w));
    }

    return wrote;
}
