#include <SFML/Graphics.hpp>
#include<string.h>
#include"bullet.h"
using namespace sf;



class AddOns{
public:
    Texture texPower, texFire, texLives, texDanger;
    Sprite spritePower, spriteFire, spriteLives, spriteDanger;
    
    AddOns() {
        // Load textures
    texPower.loadFromFile("img/power.png");
    texFire.loadFromFile("img/fire.png");
    texLives.loadFromFile("img/lives.png");
    texDanger.loadFromFile("img/danger.png");
   // Set sprites' textures and scales
    spritePower.setTexture(texPower);
    spritePower.setScale(1, 1);
    spritePower.setPosition(200,-20);
    spriteFire.setTexture(texFire);
    spriteFire.setScale(1, 1);
    spriteFire.setPosition(300,-20);
    spriteLives.setTexture(texLives);
    spriteLives.setScale(1, 1);
    spriteLives.setPosition(100,-20);
    spriteDanger.setTexture(texDanger);
    spriteDanger.setScale(0.5, 0.5);
    spriteDanger.setPosition(400,-20);
    }
    
    
};



class Player{
public:
Texture tex;
Sprite sprite;
Bullet B;
float speed=0.1;
int x,y;
int life=3;
Clock Fc;
int FR=5;
AddOns addons;
Player(std::string png_path)
{

tex.loadFromFile(png_path);
sprite.setTexture(tex);
x=340;y=700;
sprite.setPosition(x,y);
sprite.setScale(0.75,0.75);
}


void move(std::string s){
float delta_x=0,delta_y=0;
if(s=="l"){
	delta_x=-3;
	
	}
	
else if(s=="r"){
	delta_x=3;
	
	}
	
if(s=="u"){
	delta_y=-3;
	}
	
else if(s=="d"){
	delta_y=3;
	}

delta_x*=speed;
delta_y*=speed;

sprite.move(delta_x, delta_y);

//WrapAround Environment
  float width=sprite.getGlobalBounds().width;
  float height=sprite.getGlobalBounds().height;
  float x_p=sprite.getPosition().x;
  float y_p=sprite.getPosition().y;
    if(x_p+width<0){
        x_p=780;}
    else if(x_p>780){
        x_p=-width;}
    if(y_p+height<0){
        y_p=780;}
    else if (y_p>780){
        y_p=-height;}
sprite.setPosition(x_p,y_p);
  
}

void fire(){
if(Fc.getElapsedTime().asMilliseconds()>1000/FR){
B=Bullet("img/laserBlue03.png",sprite.getPosition().x+sprite.getGlobalBounds().width/2-5,sprite.getPosition().y-sprite.getGlobalBounds().height/2,-1);
Fc.restart();

}
}
};



class Fire: public AddOns{
public:



};


class Lives: public AddOns{
public:




};


class Danger: public AddOns{
public:


};
