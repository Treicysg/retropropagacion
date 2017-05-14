#println("hello world")
#Read file

function readFile()
  input = open("input.txt")
  s = readstring(input)
  print(s)
  close(input)

end

function main()
    readFile()
end

main()
