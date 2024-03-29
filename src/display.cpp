// clang-format off
#include "../include/includes.h"
#include "../include/display.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_ttf.h"
#include <unistd.h>
// clang-format on

Display::Display(int width, int height, std::string title) {
    this->width = std::max(MIN_W, std::min(width, MAX_W));
    this->height = std::max(MIN_H, std::min(height, MAX_H));
    this->title = title;
}

// 清楚屏幕使用的颜色
void Display::set_clear_color(unsigned char color[4]) {
    this->clc[0] = color[0];
    this->clc[1] = color[1];
    this->clc[2] = color[2];
    this->clc[3] = color[3];
}

void Display::clear_buffer() {
    for (int i = 0; i < this->width; i++) {
        for (int j = 0; j < this->height; j++) {
            this->buffer[i][j][0] = this->clc[3];
            this->buffer[i][j][1] = this->clc[2];
            this->buffer[i][j][2] = this->clc[1];
            this->buffer[i][j][3] = this->clc[0];
            this->depth_buffer[i][j] = 10000000000;
        }
    }
}

void Display::init() {
    // 初始化SDL库
    char buffer[1024];
    // 获取当前工作目录
    getcwd(buffer, 1024);
    std::string str(buffer);
    // std::string path = str + "/fonts/BalooTamma2-Medium.ttf";
    std::string path = str + "/fonts/MapleMono-SC-NF-Regular.ttf";

    if (TTF_Init() == -1)
        printf("TTF_Init: %s\n", TTF_GetError());

    for (int i = this->min_font_size; i <= this->max_font_size; i++) {
        TTF_Font* font = TTF_OpenFont(path.c_str(), i);
        if (!font)
            printf("TTF_OpenFont: %s\n", TTF_GetError());
        else {
            fonts[i] = font;
            // printf("Loaded %ipx font!\n",i);
        }
    }

    // 初始化SDL子系统
    SDL_Init(SDL_INIT_EVERYTHING);

    // 创建窗口
    this->window = SDL_CreateWindow(
        this->title.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        this->width,
        this->height,
        SDL_WINDOW_SHOWN);
    // 创建渲染器用于绘制窗口
    this->renderer = SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
    // 创建纹理用作帧缓冲区
    this->Frame = SDL_CreateTexture(this->renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, this->width, this->height);
}

void Display::show() {
    SDL_RenderPresent(this->renderer);
}

void Display::toggle_depth_buffer() {
    this->depth_buffering = !this->depth_buffering;
}

void Display::draw_text(std::string&& str, int x, int y, char* color, int size) {
    size = std::max(this->min_font_size, std::min(size, this->max_font_size));

    SDL_Surface* surfaceMessage;
    SDL_Texture* Message;
    const char* string = str.c_str();
    int w, h;

    // printf("%s\n %i %i %i\n",string,txt.color[0],txt.color[1],txt.color[2]);

    // 获取文本渲染后的尺寸
    TTF_SizeText(this->fonts[size], string, &w, &h);

    SDL_Color col = {static_cast<Uint8>(color[0]), static_cast<Uint8>(color[1]), static_cast<Uint8>(color[2])};
    surfaceMessage = TTF_RenderText_Blended(this->fonts[size], string, col);
    Message = SDL_CreateTextureFromSurface(this->renderer, surfaceMessage);

    // 设置文字的位置和尺寸
    SDL_Rect Message_rect;
    Message_rect.x = x;
    Message_rect.y = y;
    Message_rect.w = w;
    Message_rect.h = h;

    SDL_RenderCopy(this->renderer, Message, NULL, &Message_rect);

    SDL_FreeSurface(surfaceMessage);
    SDL_DestroyTexture(Message);
}

void Display::draw_text(const std::string& str, int x, int y, char* color, int size) {
    size = std::max(this->min_font_size, std::min(size, this->max_font_size));

    SDL_Surface* surfaceMessage;
    SDL_Texture* Message;
    const char* string = str.c_str();
    int w, h;

    // printf("%s\n %i %i %i\n",string,txt.color[0],txt.color[1],txt.color[2]);

    TTF_SizeText(this->fonts[size], string, &w, &h);

    SDL_Color col = {static_cast<Uint8>(color[0]), static_cast<Uint8>(color[1]), static_cast<Uint8>(color[2])};
    surfaceMessage = TTF_RenderText_Blended(this->fonts[size], string, col);
    Message = SDL_CreateTextureFromSurface(this->renderer, surfaceMessage);

    SDL_Rect Message_rect;
    Message_rect.x = x;
    Message_rect.y = y;
    Message_rect.w = w;
    Message_rect.h = h;

    SDL_RenderCopy(this->renderer, Message, NULL, &Message_rect);

    SDL_FreeSurface(surfaceMessage);
    SDL_DestroyTexture(Message);
}

void Display::flip_buffer() {
    unsigned char* bytes = nullptr;
    int pitch = 0;
    // 锁定纹理以获取像素数据的写入访问权限
    SDL_LockTexture(this->Frame, nullptr, (void**)(&bytes), &pitch);

    for (int y = 0; y < this->height; y++) {
        for (int x = 0; x < this->width; x++) {
            memcpy(&bytes[(y * this->width + x) * 4], this->buffer[x][y], 4);
        }
    }

    // 解锁纹理
    SDL_UnlockTexture(this->Frame);
    // 将纹理内容复制到渲染器，准备显示
    SDL_RenderCopy(this->renderer, Frame, NULL, NULL);
}

void Display::set_pixel(int x, int y, unsigned char* color, float depth) {
    if (x < 0 || x >= this->width) {
        // printf("clipped x\n");
        return;
    }
    if (y < 0 || y >= this->height) {
        // printf("clipped y\n");
        return;
    }

    // 开启深度缓冲
    if (depth_buffering) {
        float d = this->depth_buffer[x][y];
        if (depth < d) {
            this->depth_buffer[x][y] = depth;
        } else {
            return;
        }
    }

    this->buffer[x][y][0] = color[3];
    this->buffer[x][y][1] = color[2];
    this->buffer[x][y][2] = color[1];
    this->buffer[x][y][3] = color[0];
}

void Display::destroy() {
    SDL_DestroyWindow(this->window);
    SDL_Quit();
    TTF_Quit();
}
