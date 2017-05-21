workspace()
type RedNeuronal
    inputSize::Int64
    hiddenLayerSize::Int64
    outputSize::Int64
end

function readFile()
  #input is the name of the input file
  #input = open("C:/Users/Denisse/Documents/2017/IA/Tarea3/retropropagacion/input.txt")
  input = open("C:/Users/Treicy/Documents/IA/Proyecto 3/input.txt")

  Y = Int64[]
  X = Int64[]
  current_line_nmb = 0

  for line in readlines(input)
    current_line_nmb+=1
    #make sure it is not data from the next digit
    if current_line_nmb == 9

      for c in line
        if Int(c) == 49
            append!(Y,1)
        elseif Int(c) ==48
            append!(Y,0)
        end
      end
      current_line_nmb = 0
      #Clean X and Y
      X = reshape(X,1,64)
      print("Values of X:")
      print(X)
      println()
      Y = reshape(Y,1,10)
      print("Values of Y:")
      print(Y)
      println()
      X = Int64[]
      Y = Int64[]
    else
      for c in line
        if Int(c) == 49
            append!(X,1)
        elseif Int(c) ==48
            append!(X,0)
        end
      end


    end




  end

  # while !eof(input)
  #   c = read(input, Char)
  #   ##
  #   if lines == 9
  #   end
  #   if c == 10
  #     lines+=1
  #   end
  #   if Int(c) == 49
  #     append!(X,1)
  #   elseif Int(c) ==48
  #     append!(X,0)
  #   end
  #
  # end
  #println("Number of lines: ")
  #println(numbLines)
  #R = reshape(X,1,64)
  close(input)
  #return R

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
  #readFile()
  #Se prueba con el valor de 1
  #X = Int64[0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0]
  readFile()
  # X = readFile()
  # red = RedNeuronal(64, 30, 10)
  #
  # w1 = rand(red.inputSize, red.hiddenLayerSize)
  # w2 = rand(red.hiddenLayerSize, red.outputSize)
  #
  # result = forward(X, w1, w2)
  # println(result)
end


main()
