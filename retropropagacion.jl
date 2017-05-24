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
  db_index = 1
  digit_rep = Array{Any}(2)
  database = Array{Any}(10)
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
      #println()
      #print("*******Values of X:*********")
      #print(X)
      Y = reshape(Y,1,10)
      #println()
      #print("*********Values of Y:*******")
      #print(Y)
      digit_rep[1] = X
      digit_rep[2] = Y
      #println()
      #print("********Digit:***************")
      #print(digit_rep)
      #print("DB INDEX: ")
      #print(db_index)

      database[db_index] = digit_rep
      db_index+=1
      #println()
      X = Int64[]
      Y = Int64[]
      digit_rep = Array{Any}(2)
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
  #println("*********DATABASE*************")
  #println()
  #print(database)
  #database = reshape(database,1,10)
  close(input)
  #************uncomment the following line **************
  return database

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
  processedOut = Int64[]
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
  test_number = 1
  passed_tests = 0
  red = RedNeuronal(64, oculta, 10)
  w1 = rand(-3:3,red.inputSize, red.hiddenLayerSize)
  w2 = rand(-3:3,red.hiddenLayerSize, red.outputSize)

  database = readFile()
  for data in database
    number = data
    X = number[1]
    Y = number[2]

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
    #test is called to display values in console
    print("\n************Prueba ")
    print(test_number)
    print("************")
    passed_tests+=test(result,Y)
    test_number+=1
    open("C:/Users/Treicy/Documents/IA/Proyecto 3/output.txt", "a") do f
          n1, n2, n3, n4, n5, n6, n7, n8, n9, n0 = result
          write(f, "[$n1 $n2 $n3 $n4 $n5 $n6 $n7 $n8 $n9 $n0]\n")
          #writedlm(f, result)
      end


    #println(result)
  end

  # X = Int64[0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0]
  # Y =[1 0 0 0 0 0 0 0 0 0]
  println("\n+++++++++Resultado Final+++++++++")
  print("\nCasos Correctos:")
  print(passed_tests)
  print("\nCasos incorrectos:")
  print(10 - passed_tests)

end



function main()
  println()
  println()
  println()
  #database = readFile()
  train("archivo.txt", "C:/Users/Denisse/Documents/2017/IA/Tarea3/retropropagacion/output.txt", 0.5, 30, 20, 500000)
  #test()
  # for i in database
  #   number = i
  #   #println(number)
  #   X = number[1]
  #   Y = number[2]
  #   println("-------------X-------------------")
  #   println(X)
  #   println("-------------Y--------------------")
  #   println(Y)
  #   println()
  #   println()
  # end

  println()
  println()
  println()
  #readFile()
end
function test(red,casos)
  #red:resultado obtenido
  #casos:Y:resultado esperado


  red = reshape(red,1,10)


  print("\nResultado Esperado ---> ")
  print(casos)
  print("\nResultado Obtenido ---> ")
  print(red)
  if red == casos
    return 1
  else
    return 0

  end



end

main()
