/** @file  draw_graph.c
 * 
 *  @brief Given vertex coordinates, generate a bitmap image with edges.
 *
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "lodepng.h"
#include <string.h>

#define COL_GRADIENT 1
#define NO_BINS 7

long lwbin1_edge_count = 0;
long lwbin2_edge_count = 0;
long hgbin1_edge_count = 0;
long hgbin2_edge_count = 0;

typedef struct graph
{
  uint32_t n;
  int64_t  m;
  uint32_t *offsets;
  uint32_t *adj;
} graph_t;

static
double
timer()
{
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + ((1e-6)*tp.tv_usec));
#endif
}

static
void 
draw_line(int x0, int y0, int x1, int y1, unsigned char *image, unsigned width, int bin, long m)
{
  unsigned R = 0, G = 0, B = 0, A = 255;
  int color_isset = 0;
  if(COL_GRADIENT && NO_BINS > 1)
  {
    switch(bin)
    {
      case(0): if((double)(lwbin1_edge_count / m) <= 0.05)
               {
                 R = 251; G = 83; B = 73; A = 255; // Red-orange 251, 83, 73
                 lwbin1_edge_count++;
                 color_isset = 1;
                 break;
               }
      case(1): if(((double)lwbin2_edge_count / (double)m) <= 0.05)
               {
                 R = 251; G = 165; B = 4; A = 255; // Orange 251, 165, 4
                 lwbin2_edge_count++;
                 color_isset = 1;
               }
               break;
      case(6): if(((double)hgbin2_edge_count / (double)m) <= 0.05)
                {
                  R = 13; G = 152; B = 186; A = 255; // blue-green: 13, 152, 186
                  hgbin2_edge_count++;
                  color_isset = 1;
                  break;
                }
                
      case(5):  if(((double)hgbin1_edge_count / (double)m) <= 0.05)
                {
                  R = 4; G = 251; B = 4; A = 255; // green: 4, 251, 4
                  hgbin1_edge_count++;
                  color_isset = 1;
                }
    }
    if( (bin == 0 || bin == 1) && !color_isset)
    {
      bin += 2;
    }
    if( (bin == 5 || bin == 6) && !color_isset)
    {
      bin -= 2;
    }
    switch(bin)
    {
      case(2): R = 251; G = 174; B = 66; A = 255; // yellow-orange: 251, 174, 66
               break;
      case(3): R = 251; G = 251; B = 4; A = 255; // yellow: 251, 251, 4 
               break;
      case(4): R = 154; G = 251; B = 50; A = 255; // yellow-green: 154, 251, 50
               break;
    }
  }
  else if(NO_BINS <= 1)
  {
    R = 255, G = 255, B = 255;
  }
  
  /* Bresenham's line algorithm */
  /* Code from https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm */

  int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
  int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1; 
  int err = (dx>dy ? dx : -dy)/2;

  for(;;)
  {
    image[4 * width * y0 + 4 * x0 + 0] = R;
    image[4 * width * y0 + 4 * x0 + 1] = G;
    image[4 * width * y0 + 4 * x0 + 2] = B;
    image[4 * width * y0 + 4 * x0 + 3] = A;
    if (x0==x1 && y0==y1) break;
    int e2 = err;
    if (e2 >-dx) { err -= dy; x0 += sx; }
    if (e2 < dy) { err += dx; y0 += sy; }
  }

}

double length(double x1, double y1, double x2, double y2)
{
  return sqrt(pow(x2-x1,2)+pow(y2-y1,2));
}

int
main (int argc, char **argv)
{

  if (argc != 7)
  {
    fprintf(stderr, "Usage: %s <binary csr filename> <vertex coords filename> "
                    "<output png filename> <num coordinates> "
                    "<first coord> <second coord>\n", argv[0]);
    return 1;
  }

  char *input_filename = argv[1];
  int numCoords = atoi(argv[4]);
  int firstCoord = atoi(argv[5])-1;
  int secondCoord = atoi(argv[6])-1;
  srand(1);

  // read input graph file
  double start_timer = timer();  
  double start_timer_total = timer();
  FILE *infp = fopen(input_filename, "rb");
  if (infp == NULL)
  {
    fprintf(stderr, "Error: Could not open CSR file. Exiting ...\n");
    return 1;
  }
  long n, m;
  long rest[4];
  unsigned int *rowOffsets, *adj;
  fread(&n, 1, sizeof(long), infp);
  fread(&m, 1, sizeof(long), infp);
  fread(rest, 4, sizeof(long), infp);
  rowOffsets = (unsigned int *) malloc (sizeof(unsigned int) * (n+1));
  adj = (unsigned int *) malloc (sizeof(unsigned int) * m);
  assert(rowOffsets != NULL);
  assert(adj != NULL);
  fread(rowOffsets, n+1, sizeof(unsigned int), infp);
  fread(adj, m, sizeof(unsigned int), infp);
  fclose(infp);
  double end_timer = timer();
  double elt = end_timer - start_timer;
  fprintf(stderr, "CSR file read time: %9.6lf s\n", elt);

  graph_t g;
  g.n = n;
  g.m = m/2;
  g.offsets = rowOffsets;
  g.adj = adj;
  fprintf(stderr, "Num edges: %ld, num vertices: %u\n", m/2, g.n);
  start_timer = timer();  
  char *vertex_coords_filename = argv[2];
  infp = fopen(vertex_coords_filename, "r");
  if (infp == NULL)
  {
    fprintf(stderr, "Error: Could not open vertex coords file. Exiting ...\n");
    return 1;
  }

  unsigned width = 1920, height = 1080;

  double *vx = (double *) malloc(g.n * sizeof(double));
  assert(vx != NULL);
  double *vy = (double *) malloc(g.n * sizeof(double));
  assert(vy != NULL);

  double xmax = -1;
  double xmin = 1;
  double ymax = -1;
  double ymin = 1;

  char stringFormat[80], temp[80];
  strcpy(temp, "");
  for(int j=0; j < numCoords; j++)
  {
    strcat(temp, "%lf,");
  }
  strncpy(stringFormat, temp, strlen(temp)-1);

  for (uint32_t i=0; i<g.n; i++)
  {
    double x[numCoords];
    for(int j=0; j < numCoords-1; j++)
    {
      fscanf(infp, "%lf,", &x[j]);
    }
    fscanf(infp, "%lf",  &x[numCoords-1]);
    vx[i] = x[firstCoord]; vy[i] = x[secondCoord];
    if (vx[i] > xmax) xmax = vx[i];
    if (vx[i]< xmin) xmin = vx[i];
    if (vy[i] > ymax) ymax = vy[i];
    if (vy[i] < ymin) ymin = vy[i]; 
  }
  fclose(infp);
  end_timer = timer();
  elt = end_timer - start_timer;
  fprintf(stderr, "Coord file read time: %9.6lf s\n", elt);

  double aspect_ratio = (xmax-xmin)/(ymax-ymin);
  fprintf(stderr, "xmin %lf xmax %lf ymin %lf ymax %lf, aspect ratio %3.4lf\n", 
                  xmin, xmax, ymin, ymax, aspect_ratio);
  width = sqrt(100*n/aspect_ratio);
  height = aspect_ratio*width;
  fprintf(stderr, "Computed width %u, height %u\n", width, height);

  if(width > 100000 || height > 100000)
  {
    width /= 10;
    height /= 10;
  }
  else if(width > 20000 || height > 20000)
  {
    width /= 5;
    height /= 5;
  }

  fprintf(stderr, "Corrected width %u, height %u\n", width, height);
  char *out_filename = argv[3];
  start_timer = timer();  
  double xrange_inv = 1.0/(xmax-xmin);
  double yrange_inv = 1.0/(ymax-ymin);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (uint32_t i=0; i<g.n; i++)
  {
    vx[i] = (-xmin + vx[i])*xrange_inv*width;
    vy[i] = (-ymin + vy[i])*yrange_inv*height;
    if (vx[i] < 0) vx[i] = 0;
    if (vy[i] < 0) vy[i] = 0;
    /* handle cases where we might exceed bounds */
    if (vx[i] > (width-1)) vx[i] = width-1;
    if (vy[i] > (height-1)) vy[i] = height-1;
  }
  unsigned char* image = (unsigned char *)malloc(width * height * 4);
  /* white background */
  unsigned x, y;
  unsigned R, G, B, A;
  if(COL_GRADIENT)
  {
    R = G = B = 4;
    A = 255;
  }
  else
  {
    R = G = B = A = 255;
  }
  float max_edge_length = 0;
  if(COL_GRADIENT && NO_BINS > 1)
  {
    #ifdef _OPENMP
    #pragma omp parallel for reduction(max : max_edge_length)
    #endif
    for (long u=0; u<g.n; u++)
    {
      for (long j=g.offsets[u]; j<g.offsets[u+1]; j++)
      {
        long v = g.adj[j];
        float l_max = length(floor(vx[u]), floor(vy[u]), floor(vx[v]), floor(vy[v]));
        if(l_max > max_edge_length)
          max_edge_length = l_max;
      }
    }
  }
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2)
  #endif
  for (y = 0; y < height; y++)
  {
    for (x = 0; x < width; x++)
    {
      image[4 * width * y + 4 * x + 0] = R;
      image[4 * width * y + 4 * x + 1] = G;
      image[4 * width * y + 4 * x + 2] = B;
      image[4 * width * y + 4 * x + 3] = A;
    }
  }
  printf("Image generation\n");
  printf("Max edge length: %f\n", max_edge_length);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (long u=0; u<g.n; u++)
  {
    for (long j=g.offsets[u]; j<g.offsets[u+1]; j++)
    {
      long v = g.adj[j];
      float edge_len = length(floor(vx[u]), floor(vy[u]), floor(vx[v]), floor(vy[v]));
      int bin = (int)floor((edge_len / max_edge_length) * NO_BINS); 
      draw_line(floor(vx[u]), floor(vy[u]), floor(vx[v]), floor(vy[v]), image, width, bin, g.m);
    }
  }
  end_timer = timer();
  elt = end_timer - start_timer;
  fprintf(stderr, "Image array generation time: %9.6lf s\n", elt);
  start_timer = timer();  
  unsigned error = lodepng_encode32_file(out_filename, image, width, height);
  end_timer = timer();
  elt = end_timer - start_timer;
  fprintf(stderr, "Png file write time: %9.6lf s\n", elt);
  /*if there's an error, display it*/
  if (error) 
    printf("error %u: %s\n", error, lodepng_error_text(error));

  free(g.offsets);
  free(g.adj);

  free(vx);
  free(vy);

  free(image);
  end_timer = timer();
  elt = end_timer - start_timer_total;
  fprintf(stderr, "Total Draw time: %9.6lf s\n", elt);

  return 0;
}
