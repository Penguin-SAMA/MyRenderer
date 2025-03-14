#pragma once

#include "SDL2/SDL.h"
#include "SDL2/SDL_ttf.h"
#include <map>
#include <queue>
#include <string>

#define MAX_W 1920
#define MAX_H 1080
#define MIN_W 640
#define MIN_H 480

struct s_text {
    std::string text;
    char color[3];
    int x;
    int y;
    int size;
};

struct pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

class Display
{
public:
    unsigned char buffer[MAX_W][MAX_H][4]; // 颜色缓冲
    float depth_buffer[MAX_W][MAX_H];      // 深度缓冲
    int height = 0;
    int width = 0;
    int FONT_H = 30;
    std::string title;
    unsigned char clc[4] = {30, 30, 30, 255};
    SDL_Texture* Frame;     // 帧缓冲
    SDL_Window* window;     // SDL窗口
    SDL_Renderer* renderer; // SDL渲染器
    std::map<int, TTF_Font*> fonts;
    bool depth_buffering = true;
    int min_font_size = 5;
    int max_font_size = 35;

    Display(int width, int height, std::string title);
    void show();
    void clear_buffer();
    void init();
    void draw_text(const std::string& str, int x, int y, char* color, int size);
    void draw_text(std::string&& str, int x, int y, char* color, int size);
    void flip_buffer();
    void set_pixel(int x, int y, unsigned char* color, float depth);
    void destroy();
    void set_clear_color(unsigned char color[4]);
    void toggle_depth_buffer();

private:
    void prepare_buffers();
};
