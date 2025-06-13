
#include <SFML/Graphics.hpp>
#include <time.h>
#include<string.h>
#include <SFML/Audio.hpp>
#include <cstdlib>
#include<chrono>
#include <iostream>
using namespace std;
#include "player.h"
#include"Enemy.h"
const char title[] = "OOP-Project, Spring-2023";
using namespace sf;

class Game
{
public:
sf::Text scores;
	sf::Font font;
int score=0,lives=3;
bool isFiring = false;
float featuresTimer=0,FireAdon=0;
Sprite background; //Game background sprite
Texture bg_texture,tilt,tilt1,bulletFire;
int count=5,level=1,enemiesKill=0;
bool NextState=false;
Player* p; //player 
// add other game attributes
    Enemy* enemies[71];
  sf::FloatRect bounds1,bounds2,bounds3,bounds4,bounds5;   


Game()
{
int adder=0;
p=new Player("img/player_ship.png");
//p->B=new Bullet("img/enemy_laser.png");
  
    enemies[0] = new AlphaInvader(-300, 10, 0.1f, 50, "img/enemy_1.png");
    enemies[1] = new BetaInvader(-200, 10, 0.1f, 50, "img/enemy_2.png");
    enemies[2] = new AlphaInvader(-300, 100, 0.1f, 50, "img/enemy_1.png");
    enemies[3] = new BetaInvader(-200, 100, 0.1f, 50, "img/enemy_2.png");
    enemies[4] = new GammaInvader(-275, 200, 0.1f, 50, "img/enemy_3.png");
    enemies[5] = new Monster(-200, 30, 0.05f,300,"img/lit.png","img/monster1.png");
    enemies[6] = new Dragon(-1000, 0,500,"img/DragonFire.png","img/dragon.png");
    for(int i=7;i<71;i++){
    if(i==7 || i==8 || i==9){
    	enemies[i] = new AlphaInvader(-800+adder, 10, 0.1f, 50, "img/enemy_1.png");
    	adder+=100;
    	}
    else if(i==10 || i==11){
    	if(i==10)
    	adder=0;
    	 enemies[i] = new BetaInvader(-750+adder, 120, 0.1f, 50, "img/enemy_2.png");
    	 adder+=80;
    	 }
     else if(i==12)
    	  enemies[i] = new GammaInvader(-730, 250, 0.1f, 50, "img/enemy_3.png");
    	  
     else if(i==13 || i==14){
        adder=0;
     	if(i==14)
     	adder+=200;
       enemies[i] = new AlphaInvader(-400+adder, 10, 0.1f, 50, "img/enemy_1.png");
       
       }
     else if(i==15 || i==16){
     adder=0;
     if(i==16)
     	adder+=200;
     	enemies[i] = new BetaInvader(-400+adder, 200, 0.1f, 50, "img/enemy_2.png");
     	
     }
     else if(i==17)
       enemies[i] = new GammaInvader(-300, 100, 0.1f, 50, "img/enemy_3.png");
     else if(i>17 && i<=20){
     		if(i==18)
     		adder=0;
     	enemies[i] = new AlphaInvader(-800+adder, 10, 0.1f, 50, "img/enemy_1.png");
     	adder+=100;}
     	else if(i>20 && i<=24){
     		if(i==22)
     		adder=0;
     	enemies[i] = new BetaInvader(-840+adder, 130, 0.1f, 50, "img/enemy_2.png");
     	adder+=100;}
     	else if(i>24 && i<=27){
     		if(i==25)
     		adder=0;
     	enemies[i] = new GammaInvader(-815+adder, 250, 0.1f, 50, "img/enemy_3.png");
     	adder+=120;}
     	
     	else if(i>27 && i<=32){
     		if(i==28)
     		adder=0;
     	enemies[i] = new AlphaInvader(-600+adder, 10, 0.1f, 50, "img/enemy_1.png");
     	adder+=100;
     	}
     	else if(i>32 && i<=36){
     		if(i==33)
     		adder=0;
     	 enemies[i] =new BetaInvader(-560+adder, 130, 0.1f, 50, "img/enemy_2.png");
     	adder+=100;}
     	else if(i>36 && i<=40){
     		if(i==37)
     		adder=0;
     	enemies[i] = new GammaInvader(-555+adder, 250, 0.1f, 50, "img/enemy_3.png");
     	adder+=100;}
     	else if(i>40 && i<=45){
     		if(i==41)
     		adder=0;
     		if(i>43){
     		if(i==44)
     		adder=0;
     	enemies[i] = new AlphaInvader(-560+adder, 70, 0.1f, 50, "img/enemy_1.png");}
     	else
     	enemies[i] = new AlphaInvader(-600+adder, 0, 0.1f, 50, "img/enemy_1.png");
     		
     	adder+=100;}
     	else if(i>45 && i<=50){
     		if(i==46)
     		adder=0;
     		if(i>48){
		if(i==49)
		adder=0;
     	  enemies[i] =new BetaInvader(-555+adder, 230, 0.1f, 50, "img/enemy_2.png");}
     	  else
     	 enemies[i] =new BetaInvader(-590+adder, 150, 0.1f, 50, "img/enemy_2.png");
     	 
     	adder+=100;}
     	else if(i>50 && i<=55){
     		if(i==51)
     		adder=0;
     		if(i>53){
     		if(i==54)
     		adder=0;
     		enemies[i] = new GammaInvader(-555+adder, 410, 0.1f, 50, "img/enemy_3.png");}
     		else
     	     	enemies[i] = new GammaInvader(-585+adder, 320, 0.1f, 50, "img/enemy_3.png");
     		
     	adder+=100;}
     	else if(i>55 && i<=60){
     		if(i==56)
     	enemies[i] = new AlphaInvader(-500, 140, 0.1f, 50, "img/enemy_1.png");
     	if(i==57)
     	enemies[i] = new AlphaInvader(-400, 70, 0.1f, 50, "img/enemy_1.png");
     	if(i==58)
     	enemies[i] = new AlphaInvader(-300, 0, 0.1f, 50, "img/enemy_1.png");
     	if(i==59)
     	enemies[i] = new AlphaInvader(-200, 70, 0.1f, 50, "img/enemy_1.png");
     	if(i==60)
     	enemies[i] = new AlphaInvader(-100, 140, 0.1f, 50, "img/enemy_1.png");
     	}
     	else if(i>60 && i<=65){
     	if(i==61)
     	 enemies[i] =new BetaInvader(-490, 210, 0.1f, 50, "img/enemy_2.png");
     	 if(i==62)
     	 enemies[i] =new BetaInvader(-390, 280, 0.1f, 50, "img/enemy_2.png");
     	 if(i==63)
     	 enemies[i] =new BetaInvader(-290, 350, 0.1f, 50, "img/enemy_2.png");
     	 if(i==64)
     	 enemies[i] =new BetaInvader(-190, 280, 0.1f, 50, "img/enemy_2.png");
     	 if(i==65)
     	 enemies[i] =new BetaInvader(-90, 210, 0.1f, 50, "img/enemy_2.png");
     	}
     	else if(i>65 && i<=70){
     		if(i==66)
     		adder=0;
     	enemies[i] = new GammaInvader(-485+adder, 450, 0.1f, 50, "img/enemy_3.png");
     	adder+=100;}
 
    }
bg_texture.loadFromFile("img/background.jpg");
background.setTexture(bg_texture);
background.setScale(2, 1.5);


tilt.loadFromFile("img/player_ship1.png");
tilt1.loadFromFile("img/player_ship2.png");
bulletFire.loadFromFile("img/bulletfire.png");

}


bool checkCollisionBomb(){
bool Hit=false;
    // Get the bounds of the two sprites
    for(int i=0;i<71;i++){
    if(i==5 || i==6)
    	continue;
    bounds1 = p->sprite.getGlobalBounds();
    bounds2 = enemies[i]->B->sprite.getGlobalBounds();
     bounds3 = enemies[i]->sprite.getGlobalBounds();
   
  if (bounds1.intersects(bounds2)){
 
    Hit= true;
     
    }
    else if(bounds1.intersects(bounds3)){
    Hit= true;
   
    }
  
    }
    return Hit;
}


bool checkCollisionBullet(){
bool HitE=false;
    // Get the bounds of the two sprites
  
    for(int i=0;i<71;i++){
    if(i==5 || i==6)
    	continue;
     bounds4 = p->B.sprite1.getGlobalBounds();
     bounds5 = enemies[i]->sprite.getGlobalBounds();
   if(bounds4.intersects(bounds5)){
    HitE= true;
      enemies[i]->health-=50;
      
      if(i==0 || i==2 || i>=7 && i<=9 || i==13 || i==14 || i>=18 && i<=20 || i>=28 && i<=32 || i>=41 && i<=45 || i>=56 && i<=60)
      		score+=10;
      else if(i==1 || i==3 || i==10 || i==11 || i==15 || i==16 || i>=21 && i<=24 || i>=33 && i<=36 || i>=46 && i<=50 || i>=61 && i<=65)
      	score+=20;
      	else
      	score+=30;
      	char temp[256];
    sprintf(temp, "SCORE: %i", score);
    scores.setString(temp);
    
    }
  
    }
return HitE;    
}


 


void start_game()
{
    srand(time(0));
    RenderWindow window(VideoMode(780, 780), title);
    Clock clock;
    float timer=0;
    
 
 
 sf::SoundBuffer bufferTwo;
	bufferTwo.loadFromFile("sounds/shoot2.wav");
	sf::Sound line;
	line.setBuffer(bufferTwo);
 
 sf::SoundBuffer bufferThree;
	bufferThree.loadFromFile("sounds/death.wav");
	sf::Sound death;
	death.setBuffer(bufferThree);
	
	sf::Music track;
	track.openFromFile("sounds/final_sound.wav");
	track.setLoop(true);
	track.play();
 
	font.loadFromFile("img/OpenSans-Bold.ttf");
	scores.setFont(font);
	scores.setCharacterSize(20);
	scores.setFillColor(sf::Color::Red);
	scores.setPosition(20,0);
	sf::Vector2<float> score_scale(1.5f,1.5f);
	scores.setScale(score_scale);
	scores.setString("SCORE: 0");
	
	sf::Font fontlevel;
fontlevel.loadFromFile("img/OpenSans-Bold.ttf");
sf::Text levelText;
levelText.setFont(fontlevel);
levelText.setCharacterSize(20);
levelText.setFillColor(sf::Color::Red);
levelText.setPosition(20, 40);
sf::Vector2<float> level_scale(1.5f,1.5f);
levelText.setScale(level_scale);


       sf::Text livesT;
	sf::Font font2;
	font2.loadFromFile("img/OpenSans-Bold.ttf");
	livesT.setFont(font2);
	livesT.setCharacterSize(20);
	livesT.setFillColor(sf::Color::Red);
	livesT.setPosition(20,80);
	sf::Vector2<float> live_scale(1.5f,1.5f);
	livesT.setScale(live_scale);
	



    
    while (window.isOpen())
    {
        float time = clock.getElapsedTime().asSeconds(); 
        clock.restart();
        timer += time;  
        featuresTimer+=time;
        
      //  cout<<time<<endl; 
 	Event e;
        while (window.pollEvent(e))
        {  
            if (e.type == Event::Closed) // If cross/close is clicked/pressed
                window.close(); //close the game 
                if (Keyboard::isKeyPressed(Keyboard::E))
                        window.close();               	    
        }
          
	if (Keyboard::isKeyPressed(Keyboard::Left)) //If left key is pressed
            p->move("l");    // Player will move to left
	if (Keyboard::isKeyPressed(Keyboard::Right)) // If right key is pressed
            p->move("r");  //player will move to right
	if (Keyboard::isKeyPressed(Keyboard::Up)) //If up key is pressed
            p->move("u");    //playet will move upwards
	if (Keyboard::isKeyPressed(Keyboard::Down)) // If down key is pressed
            p->move("d");  //player will move downwards
        if ((Keyboard::isKeyPressed(Keyboard::Up))  && (Keyboard::isKeyPressed(Keyboard::Left))){
        	p->sprite.setTexture(tilt1);
        	
        
        }
        else if((Keyboard::isKeyPressed(Keyboard::Up))  && (Keyboard::isKeyPressed(Keyboard::Right))){
        	p->sprite.setTexture(tilt);
        
        }
        else if((Keyboard::isKeyPressed(Keyboard::Down))  && (Keyboard::isKeyPressed(Keyboard::Left))){
        	p->sprite.setTexture(tilt);
        
        }
        else if((Keyboard::isKeyPressed(Keyboard::Down))  && (Keyboard::isKeyPressed(Keyboard::Right))){
        	p->sprite.setTexture(tilt1);
        
        }
        else
        p->sprite.setTexture(p->tex);
        
       	 	
          
            
        if (Keyboard::isKeyPressed(Keyboard::Space)){
             p->fire();
             line.play();
             }
        if(p->sprite.getGlobalBounds().intersects(enemies[5]->spriteM.getGlobalBounds()) || p->sprite.getGlobalBounds().intersects(enemies[5]->spriteF.getGlobalBounds())){
        if(lives>0){
        p->sprite.setPosition(400,700);
           lives-=1;
           death.play();
           }
           
           }
           if(p->sprite.getGlobalBounds().intersects(enemies[6]->spriteD.getGlobalBounds()) || p->sprite.getGlobalBounds().intersects(enemies[6]->spriteDF.getGlobalBounds())){
        if(lives>0){
        p->sprite.setPosition(400,700);
           lives-=1;
           death.play();
           }
           
           }
        
        	
           	
	////////////////////////////////////////////////
	/////  Call your functions here            ////
	//////////////////////////////////////////////
	     
        checkCollisionBullet();
        checkCollisionBomb();
       
       
         p->B.move();
        
	if(checkCollisionBomb()) {
	p->sprite.setPosition(400,700);	
	    lives-=1;
	  death.play();
	    }
 	
	window.clear(Color::Black); //clears the screen
	window.draw(background);  // setting background
	
	window.draw(p->sprite);   // setting player on screen

	levelText.setString("LEVEL: " + std::to_string(level));
	livesT.setString("LIVES: " + std::to_string(lives));
	
	if(level==1){
	 if(timer<=5 ){
	       for (int i = 0; i < 5; i++) {
	       if(enemies[i]->health<=0){
	       enemies[i]->sprite.setPosition(5000, 5000);
	       enemies[i]->B->sprite.setPosition(5000, 5000);
	       }
	       else{       
           window.draw(enemies[i]->sprite);
           enemies[i]->MoveEnemy();
           enemies[i]->DropBomb();
  	   window.draw(enemies[i]->B->sprite);
  	     
        }
    }
    for(int i=0;i<5;i++){
    if(enemies[i]->health<=0)
    	enemiesKill++;
    	else enemiesKill=0;
    	}
    	if(enemiesKill>=5)
    	timer=6;
    	else timer=0;
}
	else if(timer>5 && timer<=15){
	enemiesKill=0;
	
        window.draw(enemies[5]->spriteM);
         enemies[5]->MoveMonster(time);
        window.draw(enemies[5]->spriteF);
        
       }
       else if(timer>15 && timer<=20){
        enemies[5]->spriteF.setPosition(5000,5000);
       for (int i = 13; i < 18; i++) {
	       if(enemies[i]->health<=0){
	       enemies[i]->sprite.setPosition(5000, 5000);
	       enemies[i]->B->sprite.setPosition(5000, 5000);
	       }
	       else{       
           window.draw(enemies[i]->sprite);
           enemies[i]->MoveEnemy();
           enemies[i]->DropBomb();
  	   window.draw(enemies[i]->B->sprite);
  	     
        }
    }
    for(int i=13;i<18;i++){
    if(enemies[i]->health<=0)
    	enemiesKill++;
    	else enemiesKill=0;
    	}
    	if(enemiesKill>=5)
    	timer=21;
    	else timer=16;
  }
   else if(timer>20 && timer<=25){
      		enemiesKill=0;
  	for (int i = 7; i < 13; i++) {
	       if(enemies[i]->health<=0){
	       enemies[i]->sprite.setPosition(5000, 5000);
	       enemies[i]->B->sprite.setPosition(5000, 5000);
	       }
	       else{       
           window.draw(enemies[i]->sprite);
           enemies[i]->MoveEnemy();
           enemies[i]->DropBomb();
  	   window.draw(enemies[i]->B->sprite);
  	     
        }
    }
    for(int i=7;i<13;i++){
    if(enemies[i]->health<=0)
    	enemiesKill++;
    	else enemiesKill=0;
    	}
    	if(enemiesKill>=6)
    	timer=26;
    	else timer=21;
  }
  
       
      else  if(timer>25 && timer<=30){
       	enemies[6]->spriteD.setPosition(250, 0);
       	
           window.draw(enemies[6]->spriteD);
           window.draw(enemies[6]->spriteDF);
           enemies[6]->DragonFire(time);
 
           }
        else if(timer>30) {
        level+=1;
        cout<<"Level: "<<level<<endl;
        timer=0;
        }
        
        }
        else if(level==2){
       
        	if(timer<=10){
        for (int i = 18; i < 28; i++) {
            if(enemies[i]->health<=0){
                enemies[i]->sprite.setPosition(5000, 5000);
                enemies[i]->B->sprite.setPosition(5000, 5000);
            }
            else{       
                window.draw(enemies[i]->sprite);
                enemies[i]->MoveEnemy();
                enemies[i]->DropBomb();
                window.draw(enemies[i]->B->sprite);
            }
        }
        for(int i=18;i<=27;i++){
            if(enemies[i]->health<=0)
                enemiesKill++;
            else 
                enemiesKill=0;
        }
        if(enemiesKill>=10)
            timer=11;
        else 
            timer=0;
    }
    else if(timer>10 && timer<=15){
	enemiesKill=0;
	
        window.draw(enemies[5]->spriteM);
         enemies[5]->MoveMonster(time);
        window.draw(enemies[5]->spriteF);
        
       }
       
    else if(timer>15 && timer<=20){
    	for (int i = 28; i < 41; i++) {
            if(enemies[i]->health<=0){
                enemies[i]->sprite.setPosition(5000, 5000);
                enemies[i]->B->sprite.setPosition(5000, 5000);
            }
            else{       
                window.draw(enemies[i]->sprite);
                enemies[i]->MoveEnemy();
                enemies[i]->DropBomb();
                window.draw(enemies[i]->B->sprite);
            }
        }
        for(int i=28;i<41;i++){
            if(enemies[i]->health<=0)
                enemiesKill++;
            else 
                enemiesKill=0;
        }
        if(enemiesKill>=13)
            timer=21;
        else 
            timer=16;
    
    }
     else  if(timer>20 && timer<=25){
       	enemies[6]->spriteD.setPosition(250, 0);
       	
           window.draw(enemies[6]->spriteD);
           window.draw(enemies[6]->spriteDF);
           enemies[6]->DragonFire(time);
 
           }
    else if(timer>25){
    level+=1;
    timer=0;
    cout<<"LEVEL: "<<level<<endl;
    
    }
    	
    }
    
    else if(level==3){
    	if(timer<=10){
    		  for (int i = 41; i < 56; i++) {
            if(enemies[i]->health<=0){
                enemies[i]->sprite.setPosition(5000, 5000);
                enemies[i]->B->sprite.setPosition(5000, 5000);
            }
            else{       
                window.draw(enemies[i]->sprite);
                enemies[i]->MoveEnemy();
                enemies[i]->DropBomb();
                window.draw(enemies[i]->B->sprite);
            }
        }
        for(int i=41;i<56;i++){
            if(enemies[i]->health<=0)
                enemiesKill++;
            else 
                enemiesKill=0;
        }
        if(enemiesKill>=15)
            timer=11;
        else 
            timer=0;
    	
    	}
    	else if(timer>10 && timer<=15){
	enemiesKill=0;
	
        window.draw(enemies[5]->spriteM);
         enemies[5]->MoveMonster(time);
        window.draw(enemies[5]->spriteF);
        
       }
       else if(timer>15 && timer<=20){
    	for (int i = 56; i < 71; i++) {
            if(enemies[i]->health<=0){
                enemies[i]->sprite.setPosition(5000, 5000);
                enemies[i]->B->sprite.setPosition(5000, 5000);
            }
            else{       
                window.draw(enemies[i]->sprite);
                enemies[i]->MoveEnemy();
                enemies[i]->DropBomb();
                window.draw(enemies[i]->B->sprite);
            }
        }
        for(int i=56;i<71;i++){
            if(enemies[i]->health<=0)
                enemiesKill++;
            else 
                enemiesKill=0;
        }
        if(enemiesKill>=15)
            timer=21;
        else 
            timer=16;
    
    }
     else  if(timer>20 && timer<=25){
       	enemies[6]->spriteD.setPosition(250, 0);
       	
           window.draw(enemies[6]->spriteD);
           window.draw(enemies[6]->spriteDF);
           enemies[6]->DragonFire(time);
 
           }
    else if(timer>25){
    level+=1;
    timer=0;
    cout<<"LEVEL: "<<level<<endl;
    window.close();
    }
    	
    
    }
    
    
    if(featuresTimer>=5 && featuresTimer<=20 ){
    
        if (p->addons.spritePower.getPosition().y > 800) {
            float x = (rand() % 700) + 100;
            p->addons.spritePower.setPosition(x, -40);
        }
        
        p->addons.spritePower.move(0, 0.05);
        window.draw(p->addons.spritePower);
}
	if(featuresTimer>20 && featuresTimer<=25)
		p->addons.spritePower.setPosition(20,900);
		
		if(featuresTimer>25 && featuresTimer<=40){
        if (p->addons.spriteFire.getPosition().y > 800) {
            float x = (rand() % 700) + 100;
            p->addons.spriteFire.setPosition(x, -40);
        }
    
        p->addons.spriteFire.move(0, 0.05);
        window.draw(p->addons.spriteFire);
}
	if(featuresTimer>40 && featuresTimer<=45)
		p->addons.spriteFire.setPosition(20,900);
		
		if(featuresTimer>45 && featuresTimer<=60){
        if (p->addons.spriteLives.getPosition().y > 800) {
            float x = (rand() % 700) + 100;
            p->addons.spriteLives.setPosition(x, -40);
        }
        p->addons.spriteLives.move(0, 0.05);
        window.draw(p->addons.spriteLives);
}

	if(featuresTimer>60 && featuresTimer<=65)
		p->addons.spriteLives.setPosition(20,900);
		
		if(featuresTimer>65 && featuresTimer<=80){
        if (p->addons.spriteDanger.getPosition().y > 800) {
            float x = (rand() % 700) + 100;
            p->addons.spriteDanger.setPosition(x, -40);
        }
    
        p->addons.spriteDanger.move(0, 0.05);
        window.draw(p->addons.spriteDanger);
}
	if(p->sprite.getGlobalBounds().intersects(p->addons.spriteDanger.getGlobalBounds())){
		p->addons.spriteDanger.setPosition(1000,0);
		lives-=1;
		death.play();	 
		}
        if(p->sprite.getGlobalBounds().intersects(p->addons.spriteFire.getGlobalBounds())){
            isFiring = true;
            }
            if(p->sprite.getGlobalBounds().intersects(p->addons.spriteLives.getGlobalBounds())){
            if(lives<3){
            p->addons.spriteLives.setPosition(1000,0);
		lives+=1;
		}
		}
        
        
       if(isFiring && FireAdon>5){
    isFiring=false;
    FireAdon=0;
}

//Change the texture to fire texture while the player is touching the Firesprite
if(isFiring) {
    FireAdon+=time;
    p->B.sprite1.setTexture(bulletFire);
    
}
        else if(checkCollisionBullet())
	{ 
	p->B.sprite1.move(-500,-500);
	}
	
	window.draw(p->B.sprite1);//setting bullet on screen
         window.draw(scores);//setting scores on screen
         // draw the text to the screen
	window.draw(levelText);
        window.draw(livesT); 
        if(featuresTimer>80){
        p->addons.spriteDanger.setPosition(20,900);
        	featuresTimer=0;
		 }
             if(lives<=0)
             window.close();
	window.display();  //Displying all the sprites
    }


}


};

