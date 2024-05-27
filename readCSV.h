#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_two_dimension(char* line, double** data, char *filename);
void print_two_dimension(double** data, int row, int col);
int get_row(char *filename);
int get_col(char *filename);
double ** readCSV(char *filename, char *line, double **data);

double **readCSV(char *filename, char *line, double **data)
{
    int row, col;
    row = get_row(filename);
    col = get_col(filename);
    data = (double **)malloc(row * sizeof(int *));
	for (int i = 0; i < row; ++i){
		data[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组 
	get_two_dimension(line, data, filename);
	// printf("row = %d\n", row);
	// printf("col = %d\n", col);
	// print_two_dimension(data, row, col);
	// printf("%f\n",data[0][0]);

	return data;
}

void get_two_dimension(char* line, double** data, char *filename)
{
	FILE* stream = fopen(filename, "r");
	int i = 0;
	while (fgets(line, 4096, stream))//逐行读取
    {
    	int j = 0;
    	char *tok;
        char* tmp = strdup(line);
        for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n")){
        	data[i][j] = atof(tok);//转换成浮点数 
		}//字符串拆分操作 
        i++;
        free(tmp);
    }
    fclose(stream);//文件打开后要进行关闭操作
}

void print_two_dimension(double** data, int row, int col)
{
	int i, j;
	for(i=0; i<row; i++){
		for(j=0; j<col; j++){
			printf("%f\t", data[i][j]);
		}
		printf("\n");
	}
}

int get_row(char *filename)
{
	char line[4096];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	while(fgets(line, 4096, stream)){
		i++;
	}
	fclose(stream);
	return i;
}

int get_col(char *filename)
{
	char line[4096];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	fgets(line, 4096, stream);
	char* token = strtok(line, ",");
	while(token){
		token = strtok(NULL, ",");
		i++;
	}
	fclose(stream);
	return i;
}

