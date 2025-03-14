#pragma once

#include "Material.h"
#include "Mesh.h"

struct face {
    struct vertex* v0;
    struct vertex* v1;
    struct vertex* v2;

    struct vertex* n0;
    struct vertex* n1;
    struct vertex* n2;
};

class Model
{
private:
    void cleanup_materials();
    void cleanup_faces();
    void cleanup_vertices();

public:
    std::vector<struct vertex> normals;
    std::vector<struct vertex> vertices;
    long polygons;
    std::map<std::string, struct mtl*> materials;
    std::string name;
    Material* material = 0;
    Mesh* mesh = 0;

    float max_x;
    float max_y;
    float max_z;

    float min_x;
    float min_y;
    float min_z;

    bool _normals;
    Model();
    ~Model();

    std::vector<struct face> faces;
    std::vector<struct mtl*> mats;

    void center_model();
    void apply_attr(Material* material);
    void apply_attr(Mesh* mesh);
    bool has_normals();
    void update_data();
    void init();
    void store_triangles();
    void store_materials();
    void normalize();
    long polys();
    long norms();
    long verts();
    long tris();
    void scale(float x, float y, float z);
    void rotate_x(float angle);
    void rotate_y(float angle);
    void rotate_z(float angle);
    void translate(float x, float y, float z);
    void apply_transform(float tm[4][4]);
    void apply_transformn(float tm[4][4]);
};
