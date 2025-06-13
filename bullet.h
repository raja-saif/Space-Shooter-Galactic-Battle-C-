#include <SFML/Graphics.hpp>
#include<string.h>
using namespace sf;
class Bullet{
public:
Texture tex1;
Sprite sprite1;
float speed;
Bullet(){}
Bullet(std::string png_path,int x,int y,float s)
{


tex1.loadFromFile(png_path);
sprite1.setTexture(tex1);

sprite1.setPosition(x,y);
sprite1.setScale(0.6,1);
speed=s;
}

void move(){
sprite1.move(0,speed*2);

}

};
