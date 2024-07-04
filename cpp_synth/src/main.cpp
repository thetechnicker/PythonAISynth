#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <bitset>
#include <math.h>
#include <signal.h>
#include <rtmidi/rtmidi_c.h>
// #include <rtmidi/RtMidi.h>

#define DEFAULT_FORIER_DEGREE 10

volatile sig_atomic_t flag = 0;

void handle_interrupt(int signal)
{
  flag = 1;
}

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

double *create_array(int n, double x, double y)
{
  double *array = (double *)malloc(n * sizeof(double));
  double step = (y - x) / (n - 1);

  for (int i = 0; i < n; i++)
  {
    array[i] = x + i * step;
  }

  return array;
}

int main()
{
  signal(SIGINT, handle_interrupt);
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
    char *input_name;
    while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL)
    {
      printf("Operation: %s\t", TF_OperationName(oper));
      const char *result = strstr(TF_OperationName(oper), "input");

      if (result != NULL)
      {
        int size = 0;
        while (TF_OperationName(oper)[size] != '\0')
        {
          size++;
        }
        input_name = (char *)malloc(size * sizeof(char *));
        memcpy(input_name, TF_OperationName(oper), size * sizeof(char *));
      }
      printf("\n");
    }

    printf("\nInput Layer: %s\n\n", input_name);
    {
      TF_Operation *input_op_test = TF_GraphOperationByName(graph, input_name);
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

    printf("\n");

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    int n = fourier_degree;
    float **notes = (float **)malloc(128 * sizeof(float *));
    for (int midi = 0; midi < 128; midi++)
    {
      float freq = 440 * pow(2, ((midi - 69) / 12));
      int x_length = (int)(44100 / freq);
      double *x = create_array(x_length, -M_1_PI, M_1_PI); // replace with your array
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

      TF_Output input_op = {TF_GraphOperationByName(graph, input_name), 0};
      TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

      TF_Tensor *output_tensor = NULL;

      if (TF_GetCode(status) != TF_OK)
      {
        fprintf(stderr, "Error running session: %s\n", TF_Message(status));
        goto exit_label;
      }

      TF_SessionRun(session, NULL, &input_op, &tensor, 1, &output_op, &output_tensor, 1, NULL, 0, NULL, status);

      // printf("\nout_tensor %u\n", output_tensor);
      if (TF_GetCode(status) != TF_OK)
      {
        fprintf(stderr, "Error running session: %s\n", TF_Message(status));
        goto exit_label;
      }

      end = clock();

      void *output_data = TF_TensorData(output_tensor);
      if (!output_data)
      {
        goto exit_label;
      }

      int num_elements = TF_TensorElementCount(output_tensor);
      notes[midi] = (float *)malloc(44100 * sizeof(float));
      for (int i = 0; i < num_elements; ++i)
      {
        printf("%f\n", ((float *)output_data)[i]);
      }
      for (int i = 0; i < 44100; ++i)
      {
        notes[midi][i] = (float)(((float *)output_data)[i % num_elements]);
      }

      printf("########\n");

      free(flat_array);
    }
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("The code took %f seconds to execute.\n", cpu_time_used);

    RtMidiInPtr midiin = rtmidi_in_create_default();
    rtmidi_open_port(midiin, 0, "My Client");

    while (!flag)
    {
      // Check for MIDI messages
      unsigned char *message;
      size_t size;
      rtmidi_in_get_message(midiin, message, &size);
      while (size > 0)
      {
        printf("ok ");
        for (size_t i = 0; i < size; i++)
        {
          printf("%x ", message[i]);
        }
        printf("\n");

        rtmidi_in_get_message(midiin, message, &size);
      }
    }
    rtmidi_close_port(midiin);
    rtmidi_in_free(midiin);

    for (int i = 0; i < 128; i++)
    {
      free(notes[i]);
    }
    free(notes);
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
