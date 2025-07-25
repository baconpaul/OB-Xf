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

#include "ScalingImageCache.h"
#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>

constexpr std::array<int, 3> ScalingImageCache::zoomLevels;

ScalingImageCache::ScalingImageCache(Utils &utilsRef) : utils(utilsRef) { setSkinDir(); }

juce::Image ScalingImageCache::getImageFor(const std::string &label, const int w, const int h)
{
    initializeImage(label);
    const int zl = zoomLevelFor(label, w, h);
    guaranteeImageFor(label, zl);
    auto &imgMap = cacheImages[label];

    if (const auto it = imgMap.find(zl); it != imgMap.end() && it->second.has_value())
        return *(it->second);

    if (const auto it100 = imgMap.find(baseZoomLevel);
        it100 != imgMap.end() && it100->second.has_value())
        return *(it100->second);

    return {};
}

void ScalingImageCache::clearCache()
{
    cacheImages.clear();
    cachePaths.clear();
    cacheSizes.clear();
}

juce::Image ScalingImageCache::initializeImage(const std::string &label)
{
    if (cachePaths.find(label) != cachePaths.end())
        return {};

    if (!skinDir.exists())
        return {};

    if (const juce::File basePath = skinDir.getChildFile(label + ".png"); basePath.existsAsFile())
    {
        cachePaths[label][baseZoomLevel] = basePath;
        for (int zl : zoomLevels)
        {
            if (zl == baseZoomLevel)
                continue;

            std::string suffix = fmt::format("@{}x", zl / baseZoomLevel);
            if (juce::File zp = skinDir.getChildFile(label + suffix + ".png"); zp.existsAsFile())
                cachePaths[label][zl] = zp;
        }

        juce::Image img = juce::ImageCache::getFromFile(basePath);
        cacheImages[label][baseZoomLevel] = img;

        if (img.isValid())
            cacheSizes[label] = {img.getWidth(), img.getHeight()};
        return img;
    }
    return {};
}

int ScalingImageCache::zoomLevelFor(const std::string &label, const int w, int /*h*/)
{
    if (cacheSizes.find(label) == cacheSizes.end())
        return baseZoomLevel;

    const double back = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->scale;
    auto base = cacheSizes[label];
    const double mu = back * (static_cast<float>(w) / static_cast<float>(base.first));

    for (const int zl : zoomLevels)
    {
        if (zl >= static_cast<int>(mu * baseZoomLevel))
        {
            if (cachePaths[label].find(zl) != cachePaths[label].end())
                return zl;
        }
    }

    for (auto it = cachePaths[label].rbegin(); it != cachePaths[label].rend(); ++it)
    {
        if (it->second.existsAsFile())
            return it->first;
    }

    return baseZoomLevel;
}

void ScalingImageCache::guaranteeImageFor(const std::string &label, const int zoomLevel)
{
    auto &imgMap = cacheImages[label];

    for (auto it = imgMap.begin(); it != imgMap.end();)
    {
        if (it->first != zoomLevel)
            it = imgMap.erase(it);
        else
            ++it;
    }

    auto &pathMap = cachePaths[label];

    if (const auto it = pathMap.find(zoomLevel); it != pathMap.end())
    {
        const juce::Image img = juce::ImageCache::getFromFile(it->second);
        imgMap[zoomLevel] = img;
    }
}

void ScalingImageCache::setSkinDir()
{
    if (const juce::File theme = utils.getCurrentThemeFolder(); theme.isDirectory())
        skinDir = theme;
}
