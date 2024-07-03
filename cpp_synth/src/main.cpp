#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <math.h>
#include <stdlib.h>

#define DEFAULT_FORIER_DEGREE 10

int fourier_degree;

void fourier_basis(double *x, int x_length, int n, double **basis)
{
  int i, j;
  for (i = 0; i < x_length; ++i)
  {
    basis[i] = (double *)malloc(2 * n * sizeof(double));
    for (j = 0; j < n; ++j)
    {
      basis[i][j] = sin((j + 1) * x[i]);
      basis[i][j + n] = cos((j + 1) * x[i]);
    }
  }
}

void fourier_test()
{

  int n = fourier_degree;
  double x[] = {0.1, 0.2, 0.3, 0.4, 0.5}; // replace with your array
  int x_length = sizeof(x) / sizeof(x[0]);
  double **basis = (double **)malloc(x_length * sizeof(double *));

  fourier_basis(x, x_length, n, basis);

  for (int i = 0; i < x_length; ++i)
  {
    printf("Basis for x[%d] = %f:\n", i, x[i]);
    for (int j = 0; j < 2 * n; ++j)
    {
      printf("%f ", basis[i][j]);
    }
    printf("\n");
  }

  // Don't forget to free the allocated memory when you're done.
  for (int i = 0; i < x_length; ++i)
  {
    free(basis[i]);
  }
  free(basis);
}

int main()
{
  // Initialize a TensorFlow session.
  TF_Status *status = TF_NewStatus();
  TF_SessionOptions *options = TF_NewSessionOptions();

  // Load the model.
  const char *saved_model_dir = "../../tmp/test";
  const char *tags = "serve"; // The default serving tag.
  int ntags = 1;
  TF_Graph *graph = TF_NewGraph();
  TF_Session *session = TF_LoadSessionFromSavedModel(options, NULL, saved_model_dir, &tags, ntags, graph, NULL, status);

  if (TF_GetCode(status) == TF_OK)
  {
    printf("Model loaded successfully.\n");

    // Assume 'graph' is your TF_Graph object
    size_t pos = 0;
    TF_Operation *oper;
    while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL)
    {
      printf("Operation: %s\n", TF_OperationName(oper));
    }
    TF_Operation *input_op = TF_GraphOperationByName(graph, "serving_default_input_1");
    TF_Output input = {input_op, 0};
    int num_dims = TF_GraphGetTensorNumDims(graph, input, status);
    int64_t *dims = (int64_t *)malloc(num_dims * sizeof(int));
    TF_GraphGetTensorShape(graph, input, dims, num_dims, status);
    for (int i = 0; i < num_dims; i++)
    {
      printf("input no %d: %ld\n", i, dims[i]);
      if (i == 1)
      {
        fourier_degree = (int)(dims[i] / 2);
      }
    }
    fourier_test();

    int n = fourier_degree;
    double x[] = {0.1, 0.2, 0.3, 0.4, 0.5}; // replace with your array
    int x_length = sizeof(x) / sizeof(x[0]);
    double **basis = (double **)malloc(x_length * sizeof(double *));

    fourier_basis(x, x_length, n, basis);

    for (int i = 0; i < x_length; ++i)
    {
      free(basis[i]);
    }
    free(basis);
  }
  else
  {
    printf("%s", TF_Message(status));
  }

  // Clean up.
  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);

  printf("\n");

  return 0;
}
