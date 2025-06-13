#include <SFML/Graphics.hpp>
#include <iostream>
#include "game.h"

class Menu {
public:
Sprite spriteM;
Texture texM;
//add menu attributes here
Menu()
{
    
    texM.loadFromFile("img/menu.jpg");
    spriteM.setTexture(texM);
    spriteM.setScale(0.8, 1.2);

//constructors body
}

    void display_menu() {
        sf::RenderWindow window(sf::VideoMode(600, 600), "Menu");
        sf::Font font;
        if (!font.loadFromFile("img/OpenSans-Bold.ttf")) {
            std::cerr << "Failed to load font" << std::endl;
            return;
        }

        sf::Text title;
        title.setFont(font);
        title.setString("Menu");
        title.setCharacterSize(50);
        title.setFillColor(sf::Color::White);
        title.setPosition(270, 50);

        sf::Text startGameText;
        startGameText.setFont(font);
        startGameText.setString("1. Start game");
        startGameText.setCharacterSize(40);
        startGameText.setFillColor(sf::Color::Red);
        startGameText.setPosition(200, 200);

        sf::Text optionsText;
        optionsText.setFont(font);
        optionsText.setString("2. Options");
        optionsText.setCharacterSize(40);
        optionsText.setFillColor(sf::Color::Red);
        optionsText.setPosition(200, 250);

        sf::Text exitText;
        exitText.setFont(font);
        exitText.setString("3. Exit");
        exitText.setCharacterSize(40);
        exitText.setFillColor(sf::Color::Red);
        exitText.setPosition(200, 300);

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();
                else if (event.type == sf::Event::KeyPressed) {
                    if (event.key.code == sf::Keyboard::S) {
                        Game g;
                        g.start_game();
                        
                    }
                    else if (event.key.code == sf::Keyboard::Q) {
                        window.close();
                        
                    }
                    
                }
            }

            window.clear(sf::Color::Black);
            window.draw(spriteM);
            window.draw(title);
            window.draw(startGameText);
            window.draw(optionsText);
            window.draw(exitText);
            window.display();
        }
    }
};

