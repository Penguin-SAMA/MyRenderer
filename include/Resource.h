#pragma once

#include <string>

#define MATERIAL 1
#define MESH 2
#define TEXTURE 3

class Resource
{
private:
    int TYPE;

public:
    Resource() {};
    virtual bool load(std::string path) { return false; };
    virtual bool reload() { return false; };
    virtual void display() {};
    virtual int type() = 0;
    virtual ~Resource() {};
};
