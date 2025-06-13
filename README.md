# Space-Shooter-Galactic-Battle-C-
A dynamic SFML-based space shooter game built using C++ and Object-Oriented Programming principles. Engage in thrilling space combat against invaders, monsters, and dragons, with wrap-around movement, power-ups, and multi-phase levels.
🎮 Gameplay Features

    Spaceship Control: Move in all directions with bullet firing, fire mode, and invincibility power-up.

    Enemies:

        Invaders: Alpha, Beta, Gamma (different bombing intervals and scores).

        Monster: Horizontal movement with lightning beams.

        Dragon: Fires directionally based on spaceship’s zone.

    Add-ons:

        Power-Up (invincibility + beam fire for 5s)

        Fire Mode (bullets become fire for 5s)

        Extra Lives (+1 life)

        Danger Sign (must dodge; cannot be destroyed)

    Levels:

        3 Levels × 3 Phases each, with increasing enemy speed and complex shapes.

    Screens:

        Game Menu, Instructions, Main Gameplay, Pause, High Scores, End Screen.

💾 File Handling

    Stores player name, badge, and high score.

    Displays top 3 players with badges during gameplay.

🌟 Bonus Features

    Collision animation between enemies to form new shapes.

    Game state save/resume feature.

    Wrap-around galaxy and dynamic enemy behavior.

🛠 Tech Stack

    C++ with SFML (Simple and Fast Multimedia Library)

    Object-Oriented Programming

    File I/O for score and state saving

🧩 Design Summary (for Design PDF)
🧱 Core Classes

    Game: Manages levels, phases, screens, and overall logic.

    Player: Stores name, score, lives, power-up status.

    Spaceship: Handles movement, bullets, fire modes, collisions.

    Enemy (Base Class): Common properties and methods.

        Invader (Alpha/Beta/Gamma): Varying bomb intervals, scores.

        Monster: Horizontal movement and lightning attacks.

        Dragon: Zone-based directional fire.

    AddOn: Handles power-up, fire, life, and danger drops.

    Bullet/FireBeam: For spaceship and enemy attacks.

    ScoreManager: Handles high scores with file I/O.

    StateManager (Bonus): Saves and resumes game state.

🔄 Inheritance & Polymorphism

    Enemy is a base class with polymorphic behavior for invader types.

    AddOn subclasses implement different effects and durations.

⏱ Timers and Randomization

    SFML clock for interval-based drops, firing, power-ups, and dragon appearance.

🧠 Design Considerations

    Emphasis on modularity, reusability, and scalability.

    Use of smart pointers or RAII for resource management.

    Observer/Event pattern for interaction between objects (optional).
