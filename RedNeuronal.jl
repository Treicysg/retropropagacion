workspace()
type Neurona
  delta::Float64
  net::Float64
  out::Float64
end

type Capa
  neuronas::Array{Neurona, 1}
end

type Red
    capas::Array{Capa,3}
end


function main()
  println()
  println()
  println()

  red = Red([[] [] []])
end


main()


X = Int64[0 0 0 1 1 0 0 0
          0 0 1 1 1 0 0 0
          0 0 0 1 1 0 0 0
          0 0 0 1 1 0 0 0
          0 0 0 1 1 0 0 0
          0 0 0 1 1 0 0 0
          0 0 0 1 1 0 0 0
          0 0 1 1 1 1 0 0;
                            0 0 1 1 1 1 0 0
                            0 1 1 1 1 1 1 0
                            0 1 1 0 0 0 1 1
                            0 0 0 0 0 1 1 0
                            0 0 0 0 1 1 0 0
                            0 0 0 1 1 0 0 0
                            0 0 1 1 1 1 1 1
                            0 1 1 1 1 1 1 1;
                                            0 1 1 1 1 1 1 1
                                            0 1 1 1 1 1 1 1
                                            0 0 0 0 0 1 1 0
                                            0 0 0 1 1 0 0 0
                                            0 0 0 1 1 1 1 0
                                            0 0 0 0 0 0 1 1
                                            0 1 1 0 0 1 1 1
                                            0 0 1 1 1 1 1 0;
0 1 1 0 1 1 0 0
0 1 1 0 1 1 0 0
0 1 1 0 1 1 0 0
0 1 1 1 1 1 1 1
0 0 0 0 1 1 0 0
0 0 0 0 1 1 0 0
0 0 0 0 1 1 0 0
0 0 0 0 1 1 0 0;
              0 1 1 1 1 1 1 1
              0 1 1 0 0 0 0 0
              0 1 1 0 0 0 0 0
              0 1 1 1 1 1 0 0
              0 0 0 0 0 1 1 0
              0 0 0 0 0 0 1 1
              0 1 1 1 1 1 1 1
              0 1 1 1 1 1 1 0;
                              0 0 0 0 1 1 0 0
                              0 0 0 1 1 0 0 0
                              0 0 1 1 0 0 0 0
                              0 1 1 0 0 0 0 0
                              0 1 1 1 1 1 1 0
                              0 1 1 0 0 0 1 1
                              0 1 1 0 0 0 1 1
                              0 0 1 1 1 1 1 0;
                                              0 1 1 1 1 1 1 1
                                              0 0 0 0 0 0 1 1
                                              0 0 0 0 0 1 1 0
                                              0 0 0 0 0 1 1 0
                                              0 0 0 0 1 1 0 0
                                              0 0 0 0 1 1 0 0
                                              0 0 0 1 1 0 0 0
                                              0 0 0 1 1 0 0 0;
                                                              0 0 1 1 1 1 1 0
                                                              0 1 1 0 0 0 1 1
                                                              0 1 1 0 0 0 1 1
                                                              0 0 1 1 1 1 1 0
                                                              0 1 1 0 0 0 1 1
                                                              0 1 1 0 0 0 1 1
                                                              0 1 1 0 0 0 1 1
                                                              0 0 1 1 1 1 1 0;
                                                                              0 0 1 1 1 1 1 0
                                                                              0 1 1 0 0 0 1 1
                                                                              0 1 1 0 0 0 1 1
                                                                              0 0 1 1 1 1 1 1
                                                                              0 0 0 0 0 1 1 0
                                                                              0 0 0 0 1 1 0 0
                                                                              0 0 0 0 1 1 0 0
                                                                              0 0 0 1 1 0 0 0;
                                                                                              0 0 1 1 1 1 1 0
                                                                                              0 1 1 0 0 0 1 1
                                                                                              0 1 1 0 0 0 1 1
                                                                                              0 1 1 0 0 0 1 1
                                                                                              0 1 1 0 0 0 1 1
                                                                                              0 1 1 0 0 0 1 1
                                                                                              0 1 1 0 0 0 1 1
                                                                                              0 0 1 1 1 1 1 0]
