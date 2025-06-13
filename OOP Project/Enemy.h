#include <SFML/Graphics.hpp>
#include <cstdlib>
#include<iostream>
#include<string.h>
#include"Bomb.h"

using namespace std;
class Enemy {
public:
float  FireT=0,DFtimer=0;
Bomb* B;
	bool mR = true, mD = true,MR=true,MD=true;
    float x, y, speed, health;
    Sprite sprite,spriteM,spriteF,spriteD,spriteDF;
    Texture tex2,texM,texF,texD,texDF;
    Enemy(float x, float y, float speed, float health, std::string png_path) {
        this->x = x;
        this->y = y;
        this->speed = speed;
        this->health = health;
        tex2.loadFromFile(png_path);
        sprite.setTexture(tex2);
        sprite.setPosition(x, y);
      
  
   B = new Bomb("img/laserRed06.png");
  
    }
    Enemy(float x, float y, float speed, float health,std::string png_pathF, std::string png_path) {
        this->x = x;
        this->y = y;
        this->speed = speed;
        this->health = health;
        texM.loadFromFile(png_path);
        spriteM.setTexture(texM);
        
        spriteM.setPosition(x, y);
        
        texF.loadFromFile(png_pathF);
        spriteF.setTexture(texF);
        spriteF.setPosition(x+70, y);
    
    }
    Enemy(float x, float y, float health,std::string png_pathF, std::string png_path) {
        this->x = x;
        this->y = y;
        this->speed = speed;
        
        texD.loadFromFile(png_path);
        spriteD.setTexture(texD);
        
        spriteD.setPosition(x, y);
        
        texDF.loadFromFile(png_pathF);
        spriteDF.setTexture(texDF);
        spriteDF.setPosition(x+130, y+150);
    
    }
   virtual void MoveEnemy() {
        // rectangle shape
        if (mR && sprite.getPosition().x < 700) {
            sprite.move(speed, 0);
        } else if (mD && sprite.getPosition().y < 350) {
            sprite.move(0, speed);
        } else if (!mR && sprite.getPosition().x > 50) {
            sprite.move(-speed, 0);
        } else if (!mD && sprite.getPosition().y > 100) {
            sprite.move(0, -speed);
        }
        if (sprite.getPosition().x >= 700 && mR) {
            mR = false;
            mD = true;
        } else if (sprite.getPosition().y >= 350 && mD) {
            mR = false;
            mD = false;
        } else if (sprite.getPosition().x <= 50 && !mR) {
            mR = true;
            mD = false;
        } else if (sprite.getPosition().y <= 100 && !mD) {
            mR = true;
            mD = true;
        }
        
    }
    virtual void MoveMonster(float time){
   }
    virtual void DragonFire(float time){
   }

  void DropBomb(){
  	float del_y=0;
static float count=700;
del_y=+1;
del_y*=B->speed;
B->sprite.move(0,del_y);
count-=0.05;
//cout<<count<<endl;
if(count<=0){
B->x = sprite.getPosition().x+sprite.getGlobalBounds().width / 2;
B->y = sprite.getPosition().y-B->sprite.getGlobalBounds().width/2;
B->sprite.setPosition(B->x,B->y);
count=700;

	}
  }
  
};

class Invader : public Enemy {
public:
    Invader(float x, float y, float speed, float health, std::string png_path) : Enemy(x, y, speed, health,  png_path) { sprite.setScale(0.5f, 0.5f);}
 /*virtual  void MoveEnemy() {
   
}*/

};

class AlphaInvader : public Invader {
public:
    AlphaInvader(float x, float y, float speed, float health, std::string png_path) : Invader(x, y, speed, health,  png_path) { sprite.setScale(0.5f, 0.5f);}
     void MoveEnemy() {
    if(mR && sprite.getPosition().x < 600){
        sprite.move(speed,0);
    }
   
    else if(!mR && sprite.getPosition().x > 50){
        sprite.move(-speed,0);
    }
   
    if(sprite.getPosition().x >= 600 && mR){
        mR=false;
        mD=true;
    }
  
    else if(sprite.getPosition().x <= 50 && !mR){
       mR=true;
       mD=false;
    }
 
}
 
};

class BetaInvader : public Invader {
public:
    BetaInvader(float x, float y, float speed, float health, std::string png_path) : Invader(x, y, speed, health,  png_path) { sprite.setScale(0.5f, 0.5f);}
     void MoveEnemy() {
    if(mR && sprite.getPosition().x < 700){
        sprite.move(speed,0);
    }
 
    else if(!mR && sprite.getPosition().x > 150){
        sprite.move(-speed,0);
    }
  
    if(sprite.getPosition().x >= 700 && mR){
        mR=false;
        mD=true;
    }
  
    else if(sprite.getPosition().x <= 150 && !mR){
       mR=true;
       mD=false;
    }
  
}
};

class GammaInvader : public Enemy {
public:
    GammaInvader(float x, float y, float speed, float health, std::string png_path) : Enemy(x, y, speed, health,  png_path) { sprite.setScale(0.5f, 0.5f);}
     void MoveEnemy() {
    if(mR && sprite.getPosition().x < 625){
        sprite.move(speed,0);
    }
  
    else if(!mR && sprite.getPosition().x > 75){
        sprite.move(-speed,0);
    }
    
    if(sprite.getPosition().x >= 625 && mR){
        mR=false;
        mD=true;
    }
   
    else if(sprite.getPosition().x <= 75 && !mR){
       mR=true;
       mD=false;
    }
   
}
  
};
class Monster:public Enemy {
public:
int fire;
float x,y;
float health;
 Monster(float x, float y, float speed, float health, std::string png_path1, std::string png_path) : Enemy(x, y, speed, health, png_path1,  png_path)
 { spriteM.setScale(0.25f, 0.25f);
 spriteF.setScale(0.1f, 1.0f);
 }
 

   
 void MoveMonster(float time){
 	if(MR && spriteM.getPosition().x < 680){
        spriteM.move(speed,0);
        FireT+=time;
        // spriteF.setPosition(spriteM.getPosition().x,spriteM.getPosition().y);
        if(FireT>=2.0){
        spriteF.move(0,speed*20);
         if(FireT>4)
        FireT=0;
        }
        else
        spriteF.setPosition(spriteM.getPosition().x+70,spriteM.getPosition().y);
    }
  
    else if(!MR && spriteM.getPosition().x > 0){
        spriteM.move(-speed*1,0);
         spriteF.setPosition(5000,5000);
         FireT+=time;
    }
    
    if(spriteM.getPosition().x >= 680 && MR){
        MR=false;
        MD=true;
    }
   
    else if(spriteM.getPosition().x <= 0 && !MR){
       MR=true;
       MD=false;
    }
   
   }
   
   
};


class Dragon:public Enemy {
public:
int fire;
float x,y;
float health;
 Dragon(float x, float y, float health, std::string png_path1, std::string png_path) : Enemy(x, y, health, png_path1,  png_path)
 { spriteD.setScale(1.25f, 1.25f);
   spriteDF.setScale(1.0f, 4.0f);
 }
 
 void DragonFire(float time){
 
 
 if(DFtimer>=2 && DFtimer<=4){
        spriteDF.move(0,0.5);
        DFtimer+=time;
        if(DFtimer>3)
        DFtimer=1;
    }
    else {
    DFtimer+=time;
    spriteDF.setPosition(spriteD.getPosition().x + 130, spriteD.getPosition().y + 100);
    }
 }
 


};









