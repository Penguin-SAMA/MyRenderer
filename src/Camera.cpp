// clang-format off
#include "../include/includes.h"
#include "../include/Camera.h"
#include "../include/vector.h"
// clang-format on

// 设置相机位置
void Camera::position(float x, float y, float z) {
    this->eye_x = x;
    this->eye_y = y;
    this->eye_z = z;
}

// 设置目标点位置
void Camera::lookAt(float x, float y, float z) {
    this->at_x = x;
    this->at_y = y;
    this->at_z = z;
}

void Camera::zoom(float val) {
    this->scale *= val;
}

void Camera::rotate_x(float deg) {
    float angle = (float)(deg * M_PI) / 180.0f;

    this->eye_x -= this->at_x;
    this->eye_y -= this->at_y;
    this->eye_z -= this->at_z;

    float newy = (this->eye_y * cos(angle)) + (this->eye_z * sin(angle));
    float newz = -(this->eye_y * sin(angle)) + (this->eye_z * cos(angle));

    this->eye_y = newy;
    this->eye_z = newz;

    this->eye_x += at_x;
    this->eye_y += at_y;
    this->eye_z += at_z;
}

void Camera::rotate_y(float deg) {
    float angle = (float)(deg * M_PI) / 180.0f;

    this->eye_x -= this->at_x;
    this->eye_y -= this->at_y;
    this->eye_z -= this->at_z;

    float newx = (this->eye_x * cos(angle)) - (this->eye_z * sin(angle));
    float newz = (this->eye_x * sin(angle)) + (this->eye_z * cos(angle));

    this->eye_x = newx;
    this->eye_z = newz;

    this->eye_x += at_x;
    this->eye_y += at_y;
    this->eye_z += at_z;
}

// 计算视图变换
void Camera::update_transform() {
    Vec3 at(this->at_x, this->at_y, this->at_z);
    Vec3 eye(this->eye_x, this->eye_y, this->eye_z);
    Vec3 up(0, 1, 0);

    Vec3 zaxis = at.res(eye).normalize();     // 前向向量
    Vec3 xaxis = up.cross(zaxis).normalize(); // 右向向量
    Vec3 yaxis = zaxis.cross(xaxis);          // 上向量

    this->transform[0][0] = xaxis.x;
    this->transform[1][0] = xaxis.y;
    this->transform[2][0] = xaxis.z;
    this->transform[3][0] = -xaxis.dot(eye);

    this->transform[0][1] = yaxis.x;
    this->transform[1][1] = yaxis.y;
    this->transform[2][1] = yaxis.z;
    this->transform[3][1] = -yaxis.dot(eye);

    this->transform[0][2] = zaxis.x;
    this->transform[1][2] = zaxis.y;
    this->transform[2][2] = zaxis.z;
    this->transform[3][2] = -zaxis.dot(eye);

    this->transform[0][3] = 0;
    this->transform[1][3] = 0;
    this->transform[2][3] = 0;
    this->transform[3][3] = 1;
}
