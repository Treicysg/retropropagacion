workspace()
type RedNeuronal
    inputSize::Int64
    hiddenLayerSize::Int64
    outputSize::Int64
end

function readFile()
  input = open("C:/Users/Denisse/Documents/2017/IA/Tarea3/retropropagacion/input.txt")
  s = readstring(input)
  print(s)
  close(input)
end

function forward(X, w1, w2)
 #propaga los inputs en la readFile
 z2 = *(X, w1)
 a2 = sigmoid(z2)
 z3 = *(a2, w2)
 yHat = sigmoid(z3)
 return yHat
end

function sigmoid(z)
  return 1.0 ./ (1.0 .+ exp(-z))
end

function main()
  println()
  println()
  println()
  #Se prueba con el valor de 1
  X = Int64[0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0]

  red = RedNeuronal(64, 30, 10)

  w1 = rand(red.inputSize, red.hiddenLayerSize)
  w2 = rand(red.hiddenLayerSize, red.outputSize)

  result = forward(X, w1, w2)
  println(result)
end


main()
