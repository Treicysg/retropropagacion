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
 global z2 = *(X, w1)
 global a2 = sigmoid(z2)
 global z3 = *(a2, w2)
 yHat = sigmoid(z3)
 return yHat
end

function sigmoid(z)
  return 1.0 ./ (1.0 .+ exp(-z))
end

function sigmoidPrime(z)
  return exp(-z) ./ ((1.0 .+ exp(-z))^2)
end

function costFunctionPrime(X, Y, w1, w2)
  yHat = forward(X, w1, w2)
  pas = -(Y-yHat)
  println(pas)
  println(sigmoidPrime(z3))
  println()
  delta3 = (pas) .* sigmoidPrime(z3)
  dJdW2 = *(a2', delta3)

  delta2 = *(*(delta3, w2'), sigmoidPrime(z2))
  dJdW1 = *(X', delta2)
  return dJdW1, dJdW2
end

function costFunction(Y, yHat)
  result = 0
  for i in Y
    for j in yHat
      result += 0.5*(i-j)^2
    end
  end
  return result
end

function main()
  println()
  println()
  println()
  #Se prueba con el valor de 1
  X = Int64[0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0]
  #resultado deseado
  Y =[1 0 0 0 0 0 0 0 0 0]

  red = RedNeuronal(64, 30, 10)

  w1 = rand(red.inputSize, red.hiddenLayerSize)

  w2 = rand(red.hiddenLayerSize, red.outputSize)

  result = forward(X, w1, w2)
  error = costFunction(Y, result)
  #println(error)
println(w2)
  println(sigmoidPrime(w2))
  dJdW1, dJdW2 = costFunctionPrime(X, Y, w1, w2)

  println(dJdW1)
  println()
  println(dJdW2)
end


main()
