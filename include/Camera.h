#pragma once

class Camera
{
public:
    float at_x = 0;
    float at_y = 0;
    float at_z = 0;

    float eye_x = 0;
    float eye_y = 0;
    float eye_z = 0;

    float scale = 1;

    float transform[4][4] = {

        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}

    };

    void position(float x, float y, float z);
    void lookAt(float x, float y, float z);
    void update_transform();
    void rotate_y(float deg);
    void rotate_x(float deg);
    void zoom(float val);
};
