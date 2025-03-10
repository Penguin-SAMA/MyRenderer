// clang-format off
#include "../include/includes.h"
#include "../include/Mesh.h"
#include "../include/Resource.h"
#include "../include/utils.h"
#include "../include/vector.h"
// clang-format on

Mesh::Mesh() {
}

// 加载.obj文件
bool Mesh::load(std::string path) {
    std::string line;
    std::ifstream mesh(path); // 打开 obj 文件

    if (mesh.is_open()) {
        Mesh* m = new Mesh();
        std::string curr_mat; // 记录当前使用的材质

        // 逐行读取obj文件
        while (std::getline(mesh, line)) {
            // 分割每一行
            std::vector<std::string> line_p;
            // printf("%s\n",line.c_str());
            if (is_empty(line))
                continue;
            _split(line, line_p);

            // 1. 切换使用的材质
            if (line_p[0] == "usemtl") { // 使用材质
                // printf("material %s\n", line_p[1]);
                if (line_p.size() > 1)
                    curr_mat = line_p[1];

                // 2. 解析顶点v
            } else if (line_p[0] == "v") { // 顶点
                this->add_vertex(
                    atof(line_p[1].c_str()),
                    atof(line_p[2].c_str()),
                    atof(line_p[3].c_str()));

                // 3. 解析面f
            } else if (line_p[0] == "f") { // 面
                std::vector<std::pair<long, long>> indices;

                // 面行：f v1/vt1/vn1  v2/vt2/vn2  v3/vt3/vn3
                for (int i = 1; i < line_p.size(); i++) {
                    std::vector<std::string> index;
                    std::string str(line_p[i]);
                    int no_del = _split(str, index, '/'); // 用'/'再拆分
                    // index[0] = vIndex  (顶点坐标)
                    // index[1] = vtIndex (纹理坐标)
                    // index[2] = vnIndex (法线索引)
                    float ind = atol(index[0].c_str());
                    float nind = atol(index[index.size() - 1].c_str());
                    if (index.size() < 2 || no_del < 2) {
                        this->has_normals = false;
                        nind = 0;
                    }
                    indices.push_back(std::make_pair(ind, nind));
                    // std::vector<std::string> face_info;
                    // printf("%s <%s>\n",line_p[i].c_str(),index[0].c_str());
                }
                indices.push_back(indices[0]);
                this->add_face(indices, curr_mat);

                // 4. 解析法线vn
            } else if (line_p[0] == "vn") { // 法线
                this->has_normals = true;
                this->add_normal(
                    atof(line_p[1].c_str()),
                    atof(line_p[2].c_str()),
                    atof(line_p[3].c_str()));
            }
        }
        mesh.close();
    } else {
        return false;
    }
    return true;
}

bool Mesh::_normals() {
    return this->has_normals;
}

bool Mesh::reload() {
    // TODO
    return false;
}

void Mesh::display() {
    // TODO
    // this->print_mesh();

    for (int i = 0; i < n_list.size(); i++) {
        n_list[i].print();
    }
}

Mesh::~Mesh() {
    // TODO
}

int Mesh::type() {
    // TODO
    return 0;
}

// 添加法线
void Mesh::add_normal(float x, float y, float z) {
    struct vertex n = {x, y, z, 0};

    // printf("->adding %f %f %f %f\n",x,y,z,1.0f);

    this->n_list.push_back(n);
}

// 法线数量
long Mesh::normals() {
    return n_list.size();
}

// 添加顶点
void Mesh::add_vertex(float x, float y, float z) {
    struct vertex v = {x, y, z, 1};
    // printf("(%f %f %f %f)\n",x,y,z,1.0f);
    this->v_list.push_back(v);
}

// 顶点数量
long Mesh::vertices() {
    return v_list.size();
}

// 添加面
void Mesh::add_face(std::vector<std::pair<long, long>> face, std::string m) {
    this->f_list.push_back(face);
    // printf("face ");
    // for (int i = 0; i < face.size(); i++) {
    // printf("%i ",face[i]);
    //}
    // printf("\n");
    this->mat_list.push_back(m);
    // printf("size f %d\n",this->mat_list.size());
}

// 面数量
long Mesh::polygons() {
    return f_list.size();
}
