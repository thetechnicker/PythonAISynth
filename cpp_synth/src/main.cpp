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

    {
      TF_Operation *input_op_test = TF_GraphOperationByName(graph, "serving_default_inputs");
      TF_Output input = {input_op_test, 0};
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
    }

    fourier_test();

    printf("\n");

    int n = fourier_degree;
    double x[] = {0.1, 0.2, 0.3, 0.4, 0.5}; // replace with your array
    int x_length = sizeof(x) / sizeof(x[0]);
    double **basis = (double **)malloc(x_length * sizeof(double *));

    fourier_basis(x, x_length, n, basis);
    int y_length = 2 * n;

    float *flat_array = (float *)malloc(x_length * y_length * sizeof(float));
    for (int i = 0; i < x_length; ++i)
    {
      for (int j = 0; j < y_length; ++j)
      {
        flat_array[i * y_length + j] = (float)basis[i][j];
      }
    }
    for (int i = 0; i < x_length; ++i)
    {
      free(basis[i]);
    }
    free(basis);

    int64_t dims[] = {x_length, y_length};
    size_t num_bytes = x_length * y_length * sizeof(float);
    TF_Tensor *tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
    memcpy(TF_TensorData(tensor), flat_array, num_bytes);

    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_inputs"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

    TF_Tensor *output_tensor = NULL;

    TF_SessionRun(session, NULL, &input_op, &tensor, 1, &output_op, &output_tensor, 1, NULL, 0, NULL, status);


    printf("\nout_tensor %d\n", output_tensor);
    if (TF_GetCode(status) != TF_OK)
    {
      fprintf(stderr, "Error running session: %s\n", TF_Message(status));
      goto exit_label;
    }
    void *output_data = TF_TensorData(output_tensor);
    // if (output_data!= NULL)
    // {
    //   printf("YEAH\n");
    // }
    // else
    // {
    //   printf("NOOO\n");
    // }
    // int num_elements = TF_TensorElementCount(output_tensor);

    // for (int i = 0; i < num_elements; ++i)
    // {
    //   printf("%f\n", output_data[i]);
    // }

    free(flat_array);
  }
  else
  {
    printf("%s", TF_Message(status));
  }

exit_label:
  // Clean up.
  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);

  printf("\n");

  return 0;
}
