#include "SoundPlayer.h"
#include <algorithm>

SoundPlayer::SoundPlayer(const std::vector<std::vector<float>> &soundData)
{
    for (const auto &data : soundData)
    {
        std::vector<sf::Int16> samples(data.size());
        std::transform(data.begin(), data.end(), samples.begin(), [](float n)
                       { return static_cast<sf::Int16>(std::max(-1.f, std::min(1.f, n)) * 32767.f); });

        sf::SoundBuffer buffer;
        if (!buffer.loadFromSamples(samples.data(), samples.size(), 2, 44100))
        {
            throw std::runtime_error("Failed to load sound data");
        }
        buffers.push_back(buffer);

        sf::Sound sound;
        sound.setBuffer(buffers.back());
        sounds.push_back(sound);
    }
}

void SoundPlayer::play(size_t index)
{
    if (index >= sounds.size())
    {
        throw std::out_of_range("Sound index out of range");
    }
    auto play = [](size_t _index, std::vector<sf::Sound> _sounds)
    {
        _sounds[_index].play();
        while (_sounds[_index].getStatus() == sf::Sound::Playing)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };
    std::thread p(play, index, sounds);
    p.detach();
}

void SoundPlayer::stop(size_t index)
{
    if (index >= sounds.size())
    {
        throw std::out_of_range("Sound index out of range");
    }
    sounds[index].stop();
}

void SoundPlayer::exit()
{
    for (int i = 0; i < sounds.size(); i++)
    {
        while (sounds[i].getStatus() == sf::Sound::Playing)
        {
            sounds[i].stop();
        }
    }
}