workspace()
type RedNeuronal
    inputSize::Int64
    hiddenLayerSize::Int64
    outputSize::Int64
end


function forward(X, w1, w2)
 #propaga los inputs en la RedNeuronal
 z2 = *(X, w1)
 global a2 = sigmoid(z2 + 1)
 z3 = *(a2, w2)
 outM = sigmoid(z3 + 1)
 return outM
end

function sigmoid(z)
  return 1.0 ./ (1.0 .+ exp(-z))
end

function backwards(X, Y, w1, w2, outM, eta)
  error_out = (outM .* (1 .- outM) .* (Y .- outM))
  global w_upd2 = (w2 .+ eta .* error_out .* a2')
  error_out2 = (a2 .*(1 .- a2)) .* (*(error_out,w2'))
  global w_upd1 = (w1 .+ eta .* error_out2 .* X')
end

function processSolution(outM)
  processedOut = []
  for i in outM
    if i <= 0.5
      push!(processedOut,0)
    else
      push!(processedOut,1)
    end
  end
  return processedOut
end

function train(input, output, eta, oculta, error, max_iter)

  bias_node = 1

  red = RedNeuronal(64, oculta, 10)
  w1 = rand(-3:3,red.inputSize, red.hiddenLayerSize)
  w2 = rand(-3:3,red.hiddenLayerSize, red.outputSize)

  X = Int64[0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0]
  Y =[1 0 0 0 0 0 0 0 0 0]

  i = 0
  outM= []
  while i < max_iter

    outM = forward(X, w1, w2)
    backwards(X, Y, w1, w2, outM, eta)
    w1 = w_upd1
    w2 = w_upd2

    i += 1
  end
  result = processSolution(outM)
  println(result)
end



function main()
  println()
  println()
  println()

  train("archivo.txt", "archivo.txt", 0.5, 30, 20, 500000)
end


main()
