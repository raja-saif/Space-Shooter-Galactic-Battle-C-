#include <SFML/Graphics.hpp>
#include <cstdlib>
#include<string.h>
using namespace sf;
class Bomb{
public:
Texture tex;
Sprite sprite;
float speed=0.1;
float x,y;
Bomb(std::string png_path)
{


tex.loadFromFile(png_path);
sprite.setTexture(tex);
x=350;y=20;
sprite.setPosition(x,y);
sprite.setScale(1,0.5);

}

};


