#include <SFML/Audio.hpp>
#include <thread>
#include <vector>

class SoundPlayer {
public:
    SoundPlayer(const std::vector<std::vector<float>>& soundData);
    void play(size_t index);
    void stop(size_t index);
    void exit();

private:
    std::vector<sf::SoundBuffer> buffers;
    std::vector<sf::Sound> sounds;
};
