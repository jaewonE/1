/*

중요!! 빌드 하고자 하는 파일이 있다면 BUILD 파일에 추가해 주어야 한다.
bazel run tensorflow/lite/micro/examples/hello_world:hello_world_test

make 파일을 이용해서 실행하는 방법은 모르겠음.

*/

// C/C++ 내장 헤더 파일
#include <iostream>
#include <math.h> // 수학 함수를 사용하기 위한 헤더 파일, 예를 들어 sin 함수

// TensorFlow Lite Micro 관련 헤더 파일
#include "tensorflow/lite/core/api/error_reporter.h"           // 에러 리포터 헤더
#include "tensorflow/lite/core/c/common.h"                     // TensorFlow Lite의 C API에 대한 공통 헤더
#include "tensorflow/lite/micro/micro_interpreter.h"           // 마이크로 인터프리터 헤더
#include "tensorflow/lite/micro/micro_log.h"                   // 로깅을 위한 헤더
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"   // 연산자 해결자 헤더
#include "tensorflow/lite/micro/micro_op_resolver.h"           // 연산자 해결자 헤더
#include "tensorflow/lite/micro/micro_profiler.h"              // 프로파일러 헤더
#include "tensorflow/lite/micro/recording_micro_interpreter.h" // 기록 중인 마이크로 인터프리터 헤더
#include "tensorflow/lite/micro/system_setup.h"                // 시스템 설정 헤더
#include "tensorflow/lite/schema/schema_generated.h"           // 스키마 헤더

// TensorFlow Lite Micro의 Hello World 예제에 필요한 헤더 파일들
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h" // 부동소수점 모델 데이터
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_int8_model_data.h" // 정수형(8비트) 모델 데이터

using std::cout;
using std::endl;
namespace { // 시작: 익명 네임스페이스 (이 네임스페이스 내의 식별자는 이 파일 내에서만 접근 가능)

const tflite::Model *model = nullptr;            // 모델 객체 생성
tflite::MicroInterpreter *interpreter = nullptr; // 마이크로 인터프리터 객체 생성
TfLiteTensor *input = nullptr;                   // 입력 텐서 객체 생성
TfLiteTensor *output = nullptr;                  // 출력 텐서 객체 생성
int inference_count = 0;                         // 추론 횟수를 저장할 변수 생성

using OpResolver = tflite::MicroMutableOpResolver<1>;
} // namespace

void setup() {
  model = tflite::GetModel(g_hello_world_int8_model_data); // 모델 객체 생성
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    cout << "Model provided is schema version " << model->version() << "not equal to supported version "
         << TFLITE_SCHEMA_VERSION << endl;
    return;
  }

  /* static tflite::ops::micro::AllOpsResolver resolver; */
  OpResolver op_resolver;
  TfLiteStatus op_resolver_status = op_resolver.AddFullyConnected();
  if (op_resolver_status != kTfLiteOk) {
    cout << "AddFullyConnected failed" << endl;
    return;
  }

  /*
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
  */
  constexpr int kTensorArenaSize = 3000;  // 텐서를 저장하기 위한 메모리 영역 크기 설정
  uint8_t tensor_arena[kTensorArenaSize]; // 텐서 메모리 영역 배열 선언
  static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  /*
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }
  */
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    cout << "AllocateTensors() failed" << endl;
    return;
  } // TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;

  return;
}

void loop() {
  // 원래 examples/hello_world/constants.h 와 constant.cc에 포함되어 있던 코드
  constexpr float kXrange = 2.f * 3.14159265359f;
  constexpr int kInferencesPerCycle = 20;

  float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
  float x_val = position * kXrange;

  // Place our calculated x value in the model's input tensor
  input->data.f[0] = x_val;

  cout << "x_val: " << x_val << endl;

  // Run inference, and report any error
  // try {
  //   TfLiteStatus invoke_status = interpreter->Invoke(); // 여기에서 에러남.
  //   cout << "invoke_status: " << invoke_status << endl;
  // } catch (std::exception &e) {
  //   cout << "exception: " << e.what() << endl;
  // }
  // cout << "HERE!" << endl;

  TF_LITE_ENSURE_STATUS(interpreter->Invoke());
  TfLiteStatus invoke_status = interpreter->Invoke();
  cout << "invoke_status: " << invoke_status << endl;
  if (invoke_status != kTfLiteOk) {
    cout << "Invoke failed on x_val: " << static_cast<double>(x_val) << endl;
    return;
  }

  // Read the predicted y value from the model's output tensor
  cout << output->data.f[0] << endl;
  float y_val = output->data.f[0];

  cout << "y_val: " << y_val << endl;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  /* HandleOutput(error_reporter, x_val, y_val); */
  cout << "x_value: " << static_cast<double>(x_val) << ", y_value: " << static_cast<double>(y_val) << endl;

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle)
    inference_count = 0;
}

int main(int argc, char *argv[]) {
  setup();
  /*  while (true) {
      loop();
    }
  */
  for (int i = 0; i < 100000; i++) {
    cout << "loop[" << i << "]" << endl;
    loop();
  }
  cout << "Done" << endl;
  return 0;
}
