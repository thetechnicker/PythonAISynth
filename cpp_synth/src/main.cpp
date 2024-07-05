#include <cmath>
#include <ctime>
#include <iostream>
#include <bitset>
#include <csignal>
#include <rtmidi/rtmidi_c.h>
#include <tensorflow/c/c_api.h>
#include <vector>
#include <SoundPlayer.h>

#define DEFAULT_FORIER_DEGREE 10
#define SAMPLE_RATE 44100

std::sig_atomic_t flag = 0;

void handle_interrupt(int signal)
{
  flag = 1;
}

int fourier_degree;

void fourier_basis(double *x, int x_length, int n, double **basis)
{
  for (int i = 0; i < x_length; ++i)
  {
    basis[i] = new double[2 * n];
    for (int j = 0; j < n; ++j)
    {
      basis[i][j] = sin((j + 1) * x[i]);
      basis[i][j + n] = cos((j + 1) * x[i]);
    }
  }
}

double *create_array(int n, double x, double y)
{
  double *array = new double[n];
  double step = (y - x) / (n - 1);

  for (int i = 0; i < n; i++)
  {
    array[i] = x + i * step;
  }

  return array;
}

int main()
{
  std::signal(SIGINT, handle_interrupt);
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
    std::cout << "Model loaded successfully.\n";

    // Assume 'graph' is your TF_Graph object
    size_t pos = 0;
    TF_Operation *oper;
    std::string input_name;
    while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL)
    {
      std::cout << "Operation: " << TF_OperationName(oper) << "\t";
      const char *result = strstr(TF_OperationName(oper), "input");

      if (result != NULL)
      {
        input_name = TF_OperationName(oper);
      }
      std::cout << "\n";
    }

    std::cout << "\nInput Layer: " << input_name << "\n\n";
    {
      TF_Operation *input_op_test = TF_GraphOperationByName(graph, input_name.c_str());
      TF_Output input = {input_op_test, 0};
      int num_dims = TF_GraphGetTensorNumDims(graph, input, status);
      int64_t *dims = new int64_t[num_dims];
      TF_GraphGetTensorShape(graph, input, dims, num_dims, status);
      for (int i = 0; i < num_dims; i++)
      {
        std::cout << "input no " << i << ": " << dims[i] << "\n";
        if (i == 1)
        {
          fourier_degree = static_cast<int>(dims[i] / 2);
        }
      }
      delete[] dims;
    }

    std::cout << "\n";

    SoundPlayer *sounds;
    {
      std::clock_t start, end;
      double cpu_time_used;
      start = std::clock();

      int n = fourier_degree;
      std::vector<std::vector<float>> notes(128, std::vector<float>(SAMPLE_RATE));

      for (int midi = 0; midi < 128; midi++)
      {
        float freq = 440 * std::pow(2, ((midi - 69) / 12.0));
        int x_length = static_cast<int>(SAMPLE_RATE / freq);
        double *x = create_array(x_length, -M_1_PI, M_1_PI); // replace with your array
        double **basis = new double *[x_length];

        fourier_basis(x, x_length, n, basis);
        int y_length = 2 * n;

        float *flat_array = new float[x_length * y_length];
        for (int i = 0; i < x_length; ++i)
        {
          for (int j = 0; j < y_length; ++j)
          {
            flat_array[i * y_length + j] = static_cast<float>(basis[i][j]);
          }
        }
        for (int i = 0; i < x_length; ++i)
        {
          delete[] basis[i];
        }
        delete[] basis;

        int64_t dims[] = {x_length, y_length};
        size_t num_bytes = x_length * y_length * sizeof(float);
        TF_Tensor *tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
        memcpy(TF_TensorData(tensor), flat_array, num_bytes);

        TF_Output input_op = {TF_GraphOperationByName(graph, input_name.c_str()), 0};
        TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

        TF_Tensor *output_tensor = NULL;

        if (TF_GetCode(status) != TF_OK)
        {
          std::cerr << "Error running session: " << TF_Message(status) << "\n";
          goto exit_label;
        }

        TF_SessionRun(session, NULL, &input_op, &tensor, 1, &output_op, &output_tensor, 1, NULL, 0, NULL, status);

        if (TF_GetCode(status) != TF_OK)
        {
          std::cerr << "Error running session: " << TF_Message(status) << "\n";
          goto exit_label;
        }

        end = std::clock();

        void *output_data = TF_TensorData(output_tensor);
        if (!output_data)
        {
          goto exit_label;
        }

        int num_elements = TF_TensorElementCount(output_tensor);
        for (int i = 0; i < num_elements; ++i)
        {
          std::cout << static_cast<float *>(output_data)[i] << "\n";
        }
        for (int i = 0; i < SAMPLE_RATE; ++i)
        {
          notes[midi][i] = static_cast<float *>(output_data)[i % num_elements];
        }

        std::cout << "########\n";

        delete[] flat_array;
      }
      cpu_time_used = static_cast<double>(end - start) / CLOCKS_PER_SEC;
      std::cout << "The code took " << cpu_time_used << " seconds to execute.\n";

      sounds = new SoundPlayer(notes);
    }

    RtMidiInPtr midiin = rtmidi_in_create_default();
    rtmidi_open_port(midiin, 0, "Midi Through:Midi Through Port");

    while (!flag)
    {
      // Check for MIDI messages
      unsigned char message[256];
      size_t size = sizeof(message);
      ;
      double dt = rtmidi_in_get_message(midiin, message, &size);

      while (size > 0)
      {
        std::cout << "ok " << size << " ";
        for (size_t i = 0; i < size; i++)
        {
          std::cout << std::hex << static_cast<int>(message[i]) << " ";
        }
        std::cout << "\n";
        if (static_cast<int>(message[0]) == 80)
        {
          sounds->play(static_cast<int>(message[1]));
        }
        else
        {
          sounds->stop(static_cast<int>(message[1]));
        }
        rtmidi_in_get_message(midiin, message, &size);
      }
    }
    sounds->exit();
    rtmidi_close_port(midiin);
    rtmidi_in_free(midiin);
  }
  else
  {
    std::cout << TF_Message(status);
  }

exit_label:
  // Clean up.
  TF_DeleteGraph(graph);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);

  std::cout << "\n";

  return 0;
}
