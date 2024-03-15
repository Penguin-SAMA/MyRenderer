// clang-format off
#include "../include/includes.h"
#include "../include/Camera.h"
#include "../include/Material.h"
#include "../include/Mesh.h"
#include "../include/Model.h"
#include "../include/display.h"
#include "../include/utils.h"
#include "../include/vector.h"
// clang-format on

#define WIN_WIDTH 1280
#define WIN_HEIGHT 720
#define RESOLUTION 1
#define WINDOW_TITLE "Penguin's Renderer"

// 颜色结构体
struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// 边缘像素结构体
struct edge_pixel {
    int x;
    struct Color c;
    float depth;
};

// 视口结构体
struct viewport {
    float x;
    float y;
    float w;
    float h;
};

// 鼠标指针坐标
int curr_mx = 0;
int curr_my = 0;
int prev_mx = 0;
int prev_my = 0;

// 变换参数
float tx = 0.0f;
float ty = 0.0f;
float tz = 0.0f;
// 平滑变换参数
float lerp_tx = 0.0f;
float lerp_ty = 0.0f;
float lerp_tz = 0.0f;

// 相机参数
float cam_x = 0.0f;
float cam_y = 0.0f;
float camxlerp = 0.0f;
float camylerp = 0.0f;
float cam_zoom = 0.9;
float cam_lerp = 0;
// 相机倾斜参数
float tilt_x = 0;
float tilt_y = 0;
float lerpty = 0;
float lerptx = 0;
float tilt = 0;

// 渲染选项开关
bool draw_lights = false;
bool no_rasterize = false;
bool draw_vertex = false;
bool backface_culling = true;
bool draw_wireframe = false;
bool ambient = true;
bool diffuse = true;
bool specular = true;
bool smooth_shading = true;
bool dark_theme = false;
bool depth_buffering = true;
bool show_materials = true;
bool model_spin = false;
bool camera_mode = false;
bool default_material = false;

Camera* camera;
int fps = 0;

std::map<std::string, long> menu_values;
struct vector2D curr_raster[(WIN_HEIGHT * 2) + (WIN_WIDTH * 2)];                   // 光栅化向量数组
struct edge_pixel edge_pixels[WIN_HEIGHT][WIN_WIDTH];                              // 边缘像素数组
unsigned char primaryColor[4] = {255, 255, 255, 255};                              // 主要颜色
const unsigned char wireframe[3] = {55, 55, 55};                                   // 线框颜色
unsigned char noMaterialColor[4] = {0, 0, 0};                                      // 无材质颜色
char text_color[3] = {120, 120, 120};                                              // 文本颜色
const unsigned char wireframe2[3] = {255, 255, 0};                                 // 第二种线框颜色（）
unsigned char wireframeColor[4] = {wireframe[0], wireframe[1], wireframe[2], 255}; // 线框颜色数组（）
unsigned char clear_color[4] = {225, 225, 225, 255};                               // 清除颜色
bool no_clipping = false;                                                          // 启用裁剪
long selected = 0;
const struct viewport vp = {150, 50, 800, 600};           // 视口设置
const float far = 10;                                     // 远平面距离
const float near = 1;                                     // 近平面距离
const int fov = 90;                                       // 视场角度
const float S = 1 / (tan((fov / 2) * (M_PI / 180)));      // 投影矩阵的缩放因子
const float aspect_ratio = (float)WIN_WIDTH / WIN_HEIGHT; // 宽高比
struct edge_pixel empty = {-1, (unsigned char)NULL, 0};   // 空边缘像素（）
std::string g_mesh_path;                                  // 模型路径
std::string g_mtl_path;                                   // 材质路径
float projection_matrix[4][4] = {                         // 投影矩阵

    {S / aspect_ratio, 0, 0, 0},
    {0, S, 0, 0},
    {0, 0, -(far / (far - near)), -1},
    {0, 0, -(far* near) / (far - near), 0}

};

std::vector<std::pair<std::string, std::pair<std::string, bool>>> menu;

// 灯光
float light_move = 0;                                                                                 // 光源移动参数
const int lights = 5;                                                                                 // 光源数量
float ambient_light[3] = {(float)(70.0f / 255.0f), (float)(70.0f / 255.0f), (float)(70.0f / 255.0f)}; // 环境光颜色
float point_light[lights][6] = {
    // 点光源设置

    {(float)(200.0f / 255.0f), (float)(200.0f / 255.0f), (float)(200.0f / 255.0f), 0, 3, 4},
    {(float)(160.0f / 255.0f), (float)(160.0f / 255.0f), (float)(160.0f / 255.0f), 0, 4, -4},
    {(float)(40.0f / 255.0f), (float)(40.0f / 255.0f), (float)(40.0f / 255.0f), 0, 10, 0},
    {(float)(160.0f / 255.0f), (float)(160.0f / 255.0f), (float)(160.0f / 255.0f), 20, 3, -4},
    {(float)(190.0f / 255.0f), (float)(190.0f / 255.0f), (float)(190.0f / 255.0f), -20, 3, -4},

};

std::vector<Model*> models;
Display* display;

// 生成边缘像素的颜色和深度信息
void get_pairs(struct vector3D poly_r[4], struct Color v_a[4], int min, int minx) {
    // 遍历三角形的每条边
    for (int k = 0; k < 3; k++) {
        // 获取当前边的顶点颜色和下一个顶点的颜色
        float r = (float)v_a[k].r;
        float g = (float)v_a[k].g;
        float b = (float)v_a[k].b;

        float r1 = (float)v_a[k + 1].r;
        float g1 = (float)v_a[k + 1].g;
        float b1 = (float)v_a[k + 1].b;

        // 获取当前边的顶点位置和下一个顶点的位置
        int x0 = (int)poly_r[k].x;
        int y0 = (int)poly_r[k].y;
        float z0 = (float)poly_r[k].z;
        int x1 = (int)poly_r[k + 1].x;
        int y1 = (int)poly_r[k + 1].y;
        float z1 = (float)poly_r[k + 1].z;

        // printf("x0 %i y0 %i x1 %i y1 %i z0 %f z1 %f\n",x0,y0,x1,y1,z0,z1);
        //  计算两点之间的x, y方向差值
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);

        // 确定线条绘制的方向
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;

        int err = dx - dy;

        // 使用Bresenham算法绘制线条
        // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        unsigned char c[4] = {wireframe[0], wireframe[1], wireframe[2], 255};
        int index = 0;
        while (true) {
            struct vector2D point = {x0, y0};
            curr_raster[index] = point;
            if ((x0 == x1) && (y0 == y1))
                break;
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
            index++;
        }

        // 计算点总数
        int no_points = index;

        // 计算颜色和深度的变化量，用于线性插值
        float delta_r = r1 - r;
        float delta_g = g1 - g;
        float delta_b = b1 - b;
        float delta_z = z1 - z0;

        float sz = (1 / z0);
        float ez = (1 / z1);
        float dz = ez - sz;
        float dz1 = 1 / dz;

        // 对每个点进行颜色和深度的插值
        for (int i = 0; i < no_points; i++) {
            float prop = (float)i / (no_points - 1);
            int x = curr_raster[i].x;
            int y = curr_raster[i].y;
            float depth = 1 / (z0 + (prop * delta_z));
            float dprop = (float)(sz * prop) / ((prop * -dz) + ez);
            struct Color col = {
                (unsigned char)(r + (dprop * delta_r)),
                (unsigned char)(g + (dprop * delta_g)),
                (unsigned char)(b + (dprop * delta_b))};
            struct edge_pixel n = {x, col, depth};
            edge_pixels[y - min][x - minx] = n;
        }
    }
}

// 使用Liang Barsky算法对三角形的顶点进行裁剪
void clip_triangle(std::vector<long>& ply, std::vector<struct vertex>& new_poly, std::vector<struct vertex>& clip_coords) {
    // 遍历顶点
    for (int j = 0; j < ply.size() - 1; j++) {
        long sv = ply[j];
        long ev = ply[j + 1];

        struct vertex* s = &clip_coords[sv - 1];
        struct vertex* e = &clip_coords[ev - 1];

        if (no_clipping) {
            struct vertex n = {e->x, e->y, e->z, e->w};
            struct vertex n1 = {s->x, s->y, s->z, s->w};
            new_poly.push_back(n);
            continue;
        }
        if (s->z > s->w && e->z > e->w)
            continue;

        // 实现Liang Barsky线段裁剪算法
        // https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm
        if (e->z <= e->w && s->z >= s->w) {
            float t = (s->w - e->z) / (s->z - e->z);
            float nx = e->x + (t * (s->x - e->x));
            float ny = e->y + (t * (s->y - e->y));
            struct vertex n = {nx, ny, s->w, s->w};
            struct vertex n1 = {e->x, e->y, e->z, e->w};
            new_poly.push_back(n);
            new_poly.push_back(n1);
        } else if (s->z <= s->w && e->z >= e->w) {
            float t = (e->w - s->z) / (e->z - s->z);
            float nx = s->x + (t * (e->x - s->x));
            float ny = s->y + (t * (e->y - s->y));
            struct vertex n = {nx, ny, e->w, e->w};
            new_poly.push_back(n);
        } else {
            struct vertex n = {e->x, e->y, e->z, e->w};
            new_poly.push_back(n);
        }
    }

    new_poly.push_back(new_poly[0]);
}

// 清除边缘像素数组的内容
void clear_edge_pixels() {
    for (int y = 0; y < WIN_HEIGHT; y++) {
        for (int x = 0; x < WIN_WIDTH; x++) {
            edge_pixels[y][x] = empty;
        }
    }
}

// 绘制点
void draw_point(int x, int y, float z, int weight) {
    for (int i = x; i < weight + x; i++) {
        for (int j = y; j < weight + y; j++) {
            display->set_pixel(i - (weight / 2), j - (weight / 2), primaryColor, z);
        }
    }
}

// 使用Bresenham算法绘制直线
void draw_line(float stx, float sty, float ex, float ey) {
    int x0 = (int)stx;
    int y0 = (int)sty;
    int x1 = (int)ex;
    int y1 = (int)ey;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);

    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;

    int err = dx - dy;
    unsigned char c[4] = {wireframe2[0], wireframe2[1], wireframe2[2], 255};
    int index = 0;
    while (true) {
        display->set_pixel(x0, y0, c, -100);
        ;
        if ((x0 == x1) && (y0 == y1))
            break;
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
        index++;
    }
}

// 应用矩阵变换到顶点
struct vertex apply_transformation(struct vertex* v, float m[4][4]) {
    float x = (v->x * m[0][0]) + (v->y * m[1][0]) + (v->z * m[2][0]) + (v->w * m[3][0]);
    float y = (v->x * m[0][1]) + (v->y * m[1][1]) + (v->z * m[2][1]) + (v->w * m[3][1]);
    float z = (v->x * m[0][2]) + (v->y * m[1][2]) + (v->z * m[2][2]) + (v->w * m[3][2]);
    float w = (v->x * m[0][3]) + (v->y * m[1][3]) + (v->z * m[2][3]) + (v->w * m[3][3]);

    struct vertex v2 = {x, y, z, w};
    return v2;
}

// 绘制两个三维向量之间的直线
void draw_vector(Vec3 v1, Vec3 v2) {
    float x = (v1.x / v1.z) * WIN_WIDTH + (WIN_WIDTH / 2);
    float y = (-v1.y / v1.z) * WIN_HEIGHT + (WIN_HEIGHT / 2);

    float x1 = (v2.x / v2.z) * WIN_WIDTH + (WIN_WIDTH / 2);
    float y1 = (-v2.y / v2.z) * WIN_HEIGHT + (WIN_HEIGHT / 2);

    draw_line(x, y, x1, y1);
}

// 三角形渲染
void render_triangle(struct vertex* clip_coords[4], std::vector<struct vector3D>* colors) {
    // 设定初始最大最小值
    int maxy = 0;
    int maxx = 0;
    int miny = 100000000;
    int minx = 100000000;
    struct vector3D poly_r[4];

    for (int k = 0; k <= 3; k++) {
        struct vertex* v = clip_coords[k];

        // 透视除法
        float x = (v->x / v->w) * WIN_WIDTH + (WIN_WIDTH / 2);
        float y = (-v->y / v->w) * WIN_HEIGHT + (WIN_HEIGHT / 2);

        // 将坐标控制在屏幕内
        x = std::max(std::min((float)WIN_WIDTH - 1, x), 0.0f);
        y = std::max(std::min((float)WIN_HEIGHT - 1, y), 0.0f);

        float z = v->w;
        struct vector3D r1 = {x, y, 1 / z};

        if (draw_vertex)
            draw_point(x, y, z, 5);

        poly_r[k] = r1;
        if (y > maxy) {
            maxy = y;
        }
        if (y < miny) {
            miny = y;
        }
        if (x > maxx) {
            maxx = x;
        }
        if (x < minx) {
            minx = x;
        }
    }

    const int rangey = maxy - miny + 1;
    const int rangex = maxx - minx + 1;

    // 根据顶点颜色或材质设置来配置顶点属性
    struct Color vertex_attributes[4];
    if (!show_materials) {
        vertex_attributes[0] = {noMaterialColor[0], noMaterialColor[1], noMaterialColor[2]};
        vertex_attributes[1] = {noMaterialColor[0], noMaterialColor[1], noMaterialColor[2]};
        vertex_attributes[2] = {noMaterialColor[0], noMaterialColor[1], noMaterialColor[2]};
        vertex_attributes[3] = {noMaterialColor[0], noMaterialColor[1], noMaterialColor[2]};
    } else if (!no_rasterize) {
        vertex_attributes[0] = {(unsigned char)colors->at(0).x, (unsigned char)colors->at(0).y, (unsigned char)colors->at(0).z};
        vertex_attributes[1] = {(unsigned char)colors->at(1).x, (unsigned char)colors->at(1).y, (unsigned char)colors->at(1).z};
        vertex_attributes[2] = {(unsigned char)colors->at(2).x, (unsigned char)colors->at(2).y, (unsigned char)colors->at(2).z};
        vertex_attributes[3] = {(unsigned char)colors->at(0).x, (unsigned char)colors->at(0).y, (unsigned char)colors->at(0).z};
    }

    // 使用多边形的边界和顶点颜色信息计算边缘像素
    get_pairs(poly_r, vertex_attributes, miny, minx);

    // 填充三角形
    for (int l = 0; l < rangey; l++) {
        int yval = l + miny;

        int smallest = 100000000;
        int largest = -100000000;
        int is = 0;
        int il = 0;
        for (int b = 0; b < rangex; b++) {
            if (edge_pixels[l][b].x == -1)
                continue;

            if (edge_pixels[l][b].x <= smallest) {
                smallest = edge_pixels[l][b].x;
                is = b;
            }
            if (edge_pixels[l][b].x >= largest) {
                largest = edge_pixels[l][b].x;
                il = b;
            }
        }
        int first = smallest;
        int last = largest;

        // 如果不进行光栅化处理，则直接在边缘像素上绘制像素点
        if (no_rasterize) {
            for (int n = first; n < last + 1; n++) {
                if (edge_pixels[l][n - minx].x != -1) {
                    display->set_pixel(n, yval, wireframeColor, -1);
                    // printf("%i %i\n",l,n-minx);
                    edge_pixels[l][n - minx] = empty;
                }
            }
            continue;
        }

        // printf("first last - > (%d, %d)\n",first,last);

        float startz = 1 / edge_pixels[l][is].depth;
        float endz = 1 / edge_pixels[l][il].depth;

        float sz = 1 / startz;
        float ez = 1 / endz;

        // 计算深度和颜色的插值步长
        float dz = ez - sz;
        float dz1 = 1 / dz;

        float startr = (float)edge_pixels[l][is].c.r;
        float startg = (float)edge_pixels[l][is].c.g;
        float startb = (float)edge_pixels[l][is].c.b;

        float endr = (float)edge_pixels[l][il].c.r;
        float endg = (float)edge_pixels[l][il].c.g;
        float endb = (float)edge_pixels[l][il].c.b;

        float delta_z = endz - startz;
        float delta_r = endr - startr;
        float delta_g = endg - startg;
        float delta_b = endb - startb;

        // printf("first last - > (%d, %d)\n",first,last);
        // printf("newline\n");

        // 对每个像素的颜色和深度进行插值计算
        for (int n = first; n < last + 1; n++) {
            float prop = (float)(n - first) / (last - first + 1);
            float z = 1 / (startz + (prop * delta_z));
            if (edge_pixels[l][n - minx].x != -1) {
                // printf("depth %f %f\n",edge_pixels[l][n-minx].depth,z);
                if (!draw_wireframe) {
                    unsigned char col[4] = {
                        edge_pixels[l][n - minx].c.r,
                        edge_pixels[l][n - minx].c.g,
                        edge_pixels[l][n - minx].c.b,
                        255};
                    display->set_pixel(n, yval, col, z);
                } else {
                    display->set_pixel(n, yval, wireframeColor, z);
                }
                edge_pixels[l][n - minx] = empty;
                continue;
            }

            float dprop = (float)(sz * prop) / ((prop * -dz) + ez);
            unsigned char r = (unsigned char)(startr + (dprop * delta_r));
            unsigned char g = (unsigned char)(startg + (dprop * delta_g));
            unsigned char b = (unsigned char)(startb + (dprop * delta_b));
            unsigned char color[4] = {r, g, b, 255};
            display->set_pixel(n, yval, color, z);
        }
    }
}

// 渲染3D模型
long _render_mesh(Model* m) {
    // 原始模型顶点
    struct vertex* v00;
    struct vertex* v10;
    struct vertex* v20;

    // 相机变换后的顶点
    struct vertex v01;
    struct vertex v11;
    struct vertex v21;

    // 透视变换后的顶点
    struct vertex v02;
    struct vertex v12;
    struct vertex v22;

    // 被剔除的三角形数
    long culled = 0;

    if (no_rasterize || !show_materials) {
        for (int i = 0; i < m->tris(); i++) {
            std::vector<struct vector3D> colors;
            struct face f = m->faces.at(i);

            v00 = f.v0;
            v10 = f.v1;
            v20 = f.v2;

            v01 = apply_transformation(v00, camera->transform);
            v11 = apply_transformation(v10, camera->transform);
            v21 = apply_transformation(v20, camera->transform);

            v02 = apply_transformation(&v01, projection_matrix);
            v12 = apply_transformation(&v11, projection_matrix);
            v22 = apply_transformation(&v21, projection_matrix);
            struct vertex* clip_coords[4] = {&v02, &v12, &v22, &v02};

            Vec3 vec0(v01.x, v01.y, v01.z);
            Vec3 vec1(v11.x, v11.y, v11.z);
            Vec3 vec2(v21.x, v21.y, v21.z);

            Vec3 res1 = vec0.res(vec1);
            Vec3 res2 = vec0.res(vec2);

            Vec3 f_norm = res1.cross(res2).normalize();
            // printf("norm -> ");
            // f_norm.print();
            Vec3 vec4(0, 0, 0);
            Vec3 diff = vec0;

            // 背面剔除
            if (backface_culling) {
                if (f_norm.dot(diff) > 0) {
                    culled++;
                    continue;
                }
            }

            if (v02.z > v02.w || v12.z > v12.w || v22.z > v22.w)
                continue;

            if (v02.z < -v02.w || v12.z < -v12.w || v22.z < -v22.w)
                continue;

            render_triangle(clip_coords, &colors);
        }
    } else {
        std::vector<Vec3> all_lights;

        // 遍历所有光源
        for (int i = 0; i < lights; i++) {
            float lx = point_light[i][3];
            float ly = point_light[i][4];
            float lz = point_light[i][5];
            float lw = 1;

            struct vertex v0 = {lx, ly, lz, lw};
            struct vertex v1 = apply_transformation(&v0, camera->transform);
            Vec3 light(v1.x, v1.y, v1.z);
            all_lights.push_back(light);

            struct vertex v2 = apply_transformation(&v1, projection_matrix);
            if (draw_lights) {
                // printf(" - %f %f\n",v2.x,v2.y);
                float _x = (v2.x / v2.w) * WIN_WIDTH + (WIN_WIDTH / 2);
                float _y = (-v2.y / v2.w) * WIN_HEIGHT + (WIN_HEIGHT / 2);
                float _z = v2.w;
                // printf("%f %f draw\n",_x,_y);
                draw_point(_x, _y, _z, 20);
            }
        }

        // 对模型中的每个三角形进行处理
        for (int i = 0; i < m->tris(); i++) {
            std::vector<struct vector3D> colors;
            struct mtl* mat = m->mats.at(i);
            if (default_material) {
                mat = new mtl();
                init_mat(mat);
            }
            bool has_normals = m->has_normals();

            struct face f = m->faces.at(i);

            struct vertex* v00 = f.v0;
            struct vertex* v10 = f.v1;
            struct vertex* v20 = f.v2;

            struct vertex* n00 = f.n0;
            struct vertex* n10 = f.n1;
            struct vertex* n20 = f.n2;

            std::vector<struct vertex> normalv;
            // n00->print();
            // n10->print();
            // n20->print();

            v01 = apply_transformation(v00, camera->transform);
            v11 = apply_transformation(v10, camera->transform);
            v21 = apply_transformation(v20, camera->transform);

            if (has_normals && smooth_shading) {
                normalv.push_back(apply_transformation(n00, camera->transform));
                normalv.push_back(apply_transformation(n10, camera->transform));
                normalv.push_back(apply_transformation(n20, camera->transform));
            }

            // 三角形顶点向量
            Vec3 vec0(v01.x, v01.y, v01.z);
            Vec3 vec1(v11.x, v11.y, v11.z);
            Vec3 vec2(v21.x, v21.y, v21.z);

            Vec3 res1 = vec0.res(vec1);
            Vec3 res2 = vec0.res(vec2);

            Vec3 f_norm = res1.cross(res2).normalize();
            // printf("norm -> ");
            // f_norm.print();

            // 创建原点向量
            Vec3 vec4(0, 0, 0);
            Vec3 diff = vec0;

            // 背面剔除
            if (backface_culling) {
                if (f_norm.dot(diff) > 0) {
                    culled++;
                    continue;
                }
            }

            // 光照计算
            std::vector<Vec3> tvs = {vec0, vec1, vec2};

            for (int j = 0; j < tvs.size(); j++) {
                float fill_r = 255.0f;
                float fill_g = 255.0f;
                float fill_b = 255.0f;

                float r = 0;
                float g = 0;
                float b = 0;
                float shininess = mat->Ns; // 高光系数

                // 环境光
                if (ambient) {
                    r += (ambient_light[0] * mat->ka[0]);
                    g += (ambient_light[1] * mat->ka[1]);
                    b += (ambient_light[2] * mat->ka[2]);
                }

                // 如果启用平滑着色并且模型包含法线信息，就用顶点的法线
                Vec3 norm;
                if (has_normals && smooth_shading) {
                    Vec3 new_n(normalv[j].x, normalv[j].y, normalv[j].z);
                    norm = new_n;
                }
                // 否则使用三角形的法线
                else {
                    norm = f_norm;
                }
                Vec3 ggnorm = norm.normalize();
                // printf("ggnorm -> ");
                // ggnorm.print();

                for (int k = 0; k < all_lights.size(); k++) {
                    Vec3 curr_v = tvs[j];
                    Vec3 light = all_lights[k];
                    Vec3 to_light = curr_v.res(light);
                    Vec3 l = to_light.normalize();

                    // 漫反射
                    if (diffuse) {
                        float h = std::max(l.dot(ggnorm), 0.0f);
                        r += (h * mat->kd[0] * point_light[k][0]);
                        g += (h * mat->kd[1] * point_light[k][1]);
                        b += (h * mat->kd[2] * point_light[k][2]);
                    }

                    // 镜面反射
                    if (specular) {
                        Vec3 lr = l.mul(1);
                        float h2 = lr.dot(ggnorm) * 2;
                        Vec3 ref1(h2 * ggnorm.x, h2 * ggnorm.y, h2 * ggnorm.z);
                        Vec3 reflected = lr.res(ref1);
                        Vec3 to_cam = curr_v.res(vec4).normalize();

                        float rfdot = (float)to_cam.dot(reflected);

                        float h0 = pow(std::max(rfdot, 0.0f), shininess);
                        r += (h0 * mat->ks[0] * point_light[k][0]);
                        g += (h0 * mat->ks[1] * point_light[k][1]);
                        b += (h0 * mat->ks[2] * point_light[k][2]);
                    }
                }

                fill_r = std::min(r * 255.0f, 255.0f);
                fill_g = std::min(g * 255.0f, 255.0f);
                fill_b = std::min(b * 255.0f, 255.0f);

                struct vector3D col = {fill_r, fill_g, fill_b};
                colors.push_back(col);
            }
            //--------

            v02 = apply_transformation(&v01, projection_matrix);
            v12 = apply_transformation(&v11, projection_matrix);
            v22 = apply_transformation(&v21, projection_matrix);
            struct vertex* clip_coords[4] = {&v02, &v12, &v22, &v02};

            // 检查变换后的顶点是否需要被裁剪
            // 如果顶点的z坐标大于其w坐标，或者z坐标小于其w坐标的负值，说明顶点在视锥体外，需要被裁剪
            if (v02.z > v02.w || v12.z > v12.w || v22.z > v22.w) // 若顶点位于远裁剪面之外
                continue;

            if (v02.z < -v02.w || v12.z < -v12.w || v22.z < -v22.w) // 若顶点位于近裁剪面之外
                continue;

            render_triangle(clip_coords, &colors);
        }
    }
    return culled;
}

// 加载模型
void load_models(std::vector<std::string> paths) {
    for (int i = 0; i < paths.size(); i++) {
        std::string mesh_path = paths[i] + ".obj";
        std::string mtl_path = paths[i] + ".mtl";

        // printf(">>%s\n",mesh_path.c_str());

        bool loaded;

        Mesh* mesh = new Mesh();
        loaded = mesh->load(mesh_path);

        if (loaded) {
            // mesh->display();
            g_mesh_path = mesh_path;
        } else {
            std::cout << "Could not load: " << mesh_path << std::endl;
        }

        Material* material = new Material();
        loaded = material->load(mtl_path);

        if (loaded) {
            // material->display();
            g_mtl_path = mtl_path;
        } else {
            std::cout << "Could not load: " << mtl_path << std::endl;
        }

        Model* model = new Model();
        model->apply_attr(mesh);
        model->apply_attr(material);

        models.push_back(model);

        // printf("model size %d\n",models.size());
    }
}

// 对模型操作
long render_mesh(Model* m) {
    cam_zoom = std::max(0.1f, cam_zoom); // 缩放值至少设置为0.1
    unsigned char color[4] = {120, 120, 120, 255};
    cam_lerp += (cam_zoom - cam_lerp) * 0.25; // 缩放插值
    float sf = cam_lerp;
    m->scale(sf, sf, sf);
    tilt_y += tilt;

    camylerp += (cam_y - camylerp) * 0.30;
    camxlerp += (cam_x - camxlerp) * 0.30;

    lerpty += (tilt_y - lerpty) * 0.30;
    lerptx += (tilt_x - lerptx) * 0.30;

    lerp_ty += (ty - lerp_ty) * 0.30;
    lerp_tx += (tx - lerp_tx) * 0.30;

    m->rotate_y(lerpty);
    m->rotate_x(lerptx);
    m->translate(lerp_tx, lerp_ty, -3);

    camera->zoom(cam_lerp);
    camera->rotate_y(camylerp);
    camera->rotate_x(camxlerp);
    camera->update_transform();
    camera->zoom(1 / cam_lerp);

    long culled = _render_mesh(m);

    camera->rotate_x(-camxlerp);
    camera->rotate_y(-camylerp);
    m->translate(-lerp_tx, -lerp_ty, 3);
    m->rotate_x(-lerptx);
    m->rotate_y(-lerpty);
    m->scale(1 / sf, 1 / sf, 1 / sf);

    return culled;
}

void render() {
    int text_size = 13;
    int margin_text = 12;
    int spacing = 3;

    menu_values["vertices"] = 0;
    menu_values["normals"] = 0;
    menu_values["polygons"] = 0;
    menu_values["triangles"] = 0;
    menu_values["triangles culled"] = 0;

    if (!no_rasterize || draw_wireframe) {
        for (int i = 0; i < models.size(); i++) {
            menu_values["vertices"] += models[i]->verts();
            menu_values["normals"] += models[i]->norms();
            menu_values["polygons"] += models[i]->polys();
            menu_values["triangles"] += models[i]->tris();
            menu_values["triangles culled"] += render_mesh(models[i]);
        }
    }

    display->draw_text("FPS " + std::to_string(fps), 10, 10, text_color, text_size);

    int i = 0;
    for (std::map<std::string, long>::iterator it = menu_values.begin(); it != menu_values.end(); ++it) {
        display->draw_text(it->first, 100, 100 + (i * (text_size + spacing)), text_color, text_size);
        display->draw_text(std::to_string(it->second), 280, 100 + (i * (text_size + spacing)), text_color, text_size);
        i++;
    }
    i = 0;
    for (int i = 0; i < menu.size(); i++) {
        bool toggled = menu[i].second.second;
        if (toggled && menu[i].first != "Toggle" && !menu[i].first.empty())
            display->draw_text("+", 100, 280 + (i * (text_size + spacing)), text_color, text_size);

        display->draw_text(menu[i].first, 100 + 15, 280 + (i * (text_size + spacing)), text_color, text_size);
        display->draw_text(menu[i].second.first, 280, 280 + (i * (text_size + spacing)), text_color, text_size);
    }

    // display->draw_text("vertices");

    display->draw_text("press (e) to reset orientation", WIN_WIDTH - 310, 100, text_color, text_size);
    display->draw_text("mtl:           " + g_mtl_path, 15, WIN_HEIGHT - 45, text_color, margin_text);
    display->draw_text("mesh:      " + g_mesh_path, 15, WIN_HEIGHT - 30, text_color, margin_text);
    display->show();
}

void update() {
}

// 暗黑主题
void update_theme() {
    if (dark_theme) {
        clear_color[0] = 30;
        clear_color[1] = 30;
        clear_color[2] = 30;
        wireframeColor[0] = 80;
        wireframeColor[1] = 80;
        wireframeColor[2] = 80;
        noMaterialColor[0] = clear_color[0] - 20;
        noMaterialColor[1] = clear_color[1] - 20;
        noMaterialColor[2] = clear_color[2] - 20;
    } else {
        clear_color[0] = 225;
        clear_color[1] = 225;
        clear_color[2] = 225;
        wireframeColor[0] = 55;
        wireframeColor[1] = 55;
        wireframeColor[2] = 55;
        noMaterialColor[0] = clear_color[0] + 20;
        noMaterialColor[1] = clear_color[1] + 20;
        noMaterialColor[2] = clear_color[2] + 20;
    }
    display->set_clear_color(clear_color);
}

// 鼠标移动事件
void handle_mouse_motion(SDL_MouseMotionEvent e) {
    if (!no_rasterize || draw_wireframe) {
        switch (e.state) {
        case SDL_BUTTON_LMASK: {
            if (camera_mode) {
                cam_y += (e.xrel * 0.25);
                cam_x += (e.yrel * 0.25);
            } else {
                tilt_y -= (e.xrel * 0.25);
                tilt_x -= (e.yrel * 0.25);
            }
            break;
        }
        case SDL_BUTTON_RMASK: {
            if (!camera_mode) {
                tx += (e.xrel * 0.005);
                ty -= (e.yrel * 0.005);
            }
            break;
        }
        }
    }
}

// 重置视角
void reset_positions() {
    tx = 0;
    ty = 0;
    tz = 0;
    tilt_x = 0;
    tilt_y = 0;
    cam_zoom = 0.9;
    cam_x = 0;
    cam_y = 0;
}

// 控制模型旋转
void handle_model_spin() {
    if (model_spin) {
        tilt = 0.9;
    } else {
        tilt = 0;
    }
}

void update_menu() {
    menu = {

        {"Toggle", {"", true}},
        {"", {"", true}},
        {"wireframe", {"w", draw_wireframe}},
        {"rasterizer", {"r", !no_rasterize}},
        {"materials", {"m", show_materials}},
        {"backface-culling", {"b", backface_culling}},
        {"ambient light", {"a", ambient}},
        {"specular light", {"s", specular}},
        {"diffuse light", {"d", diffuse}},
        {"smooth shading", {"g", smooth_shading}},
        {"depth buffering", {"z", depth_buffering}},
        {"draw lights", {"l", draw_lights}},
        {"dark theme", {"t", dark_theme}},
        {"spin model", {"c", model_spin}},
        {"camera mode", {"v", camera_mode}},
        {"default material", {"n", default_material}},
    };
}

void handle_keys(SDL_Keycode sym) {
    switch (sym) {
    case SDLK_w: {
        draw_wireframe = !draw_wireframe;
        break;
    }
    case SDLK_b: {
        backface_culling = !backface_culling;
        break;
    }
    case SDLK_r: {
        no_rasterize = !no_rasterize;
        break;
    }
    case SDLK_m: {
        show_materials = !show_materials;
        break;
    }
    case SDLK_a: {
        ambient = !ambient;
        break;
    }
    case SDLK_d: {
        diffuse = !diffuse;
        break;
    }
    case SDLK_s: {
        specular = !specular;
        break;
    }
    case SDLK_n: {
        default_material = !default_material;
        break;
    }
    case SDLK_g: {
        smooth_shading = !smooth_shading;
        break;
    }
    case SDLK_v: {
        camera_mode = !camera_mode;
        break;
    }
    case SDLK_z: {
        display->toggle_depth_buffer();
        depth_buffering = !depth_buffering;
        break;
    }
    case SDLK_l: {
        draw_lights = !draw_lights;
        break;
    }
    case SDLK_c: {
        model_spin = !model_spin;
        handle_model_spin();
        break;
    }
    case SDLK_t: {
        dark_theme = !dark_theme;
        update_theme();
        break;
    }
    case SDLK_e: {
        reset_positions();
        break;
    }
    }
    update_menu();
}

void handle_event(SDL_Event e) {
    switch (e.type) {
    case SDL_MOUSEMOTION: { // 鼠标移动事件
        handle_mouse_motion(e.motion);
        break;
    }
    case SDL_MOUSEWHEEL: { // 鼠标滚轮事件
        if (!no_rasterize || draw_wireframe)
            cam_zoom += e.wheel.y * 0.05f;
        break;
    }
    case SDL_KEYDOWN: { // 按键事件
        handle_keys(e.key.keysym.sym);
        break;
    }
    }
}

int main(int argc, char* args[]) {
    // 初始化菜单
    update_menu();

    // 初始化窗口
    display = new Display(WIN_WIDTH, WIN_HEIGHT, WINDOW_TITLE);
    display->depth_buffering = depth_buffering;
    update_theme();
    display->init();
    display->set_clear_color(clear_color);
    clear_edge_pixels();

    // 初始化相机
    camera = new Camera();
    camera->position(0, 0, 0);
    camera->lookAt(0, 0, -3);

    // 统计帧数
    int frames = 0;
    clock_t before = clock();

    // 加载模型文件
    std::vector<std::string> models;
    for (int i = 0; i < argc; i++) {
        std::vector<std::string> tmp;
        const std::string str(args[i]);
        _split(str, tmp, '.');
        if (tmp.size() > 1) {
            if (tmp[1] == "obj") {
                models.push_back(tmp[0]);
            }
        }
    }

    load_models(models);

    // SDL事件循环
    SDL_Event e;
    bool quit = false;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                quit = true;
            else
                handle_event(e);
        }
        display->clear_buffer();

        // 每秒更新帧数计数器
        if ((clock() - before) / CLOCKS_PER_SEC > 1) {
            before = clock();
            fps = frames;
            frames = 0;
        }
        update();
        render();
        display->flip_buffer();
        frames++;
    }
    display->destroy();

    return 0;
}
