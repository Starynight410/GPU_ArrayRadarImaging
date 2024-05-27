#include <stdio.h>
#include <string>
#include <stdlib.h>

int getBinSize(char *path);
void readBin(char *path, unsigned char *buf, int size);

// C 读取bin文件
int getBinSize(char *path)
{
    int  size = 0;
    FILE  *fp = fopen(path, "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    printf("path=%s,size=%d \n", path, size);
    return size;
}

void readBin(char *path, unsigned char *buf, int size)
{
    FILE *infile;
    if ((infile = fopen(path, "rb")) == NULL)
    {
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fread(buf, sizeof(char), size, infile);
    fclose(infile);
}