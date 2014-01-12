package RF

import (
	"math"
	"time"
	"math/rand"
	"fmt"
	"os"
	"encoding/json"
	"sync"
	"sync/atomic"
	"runtime"
)
type Forest struct{
	Trees []*Tree
}

// Build a new forest
func BuildForest(inputs [][]interface{},labels []string, treesAmount, samplesAmount, selectedFeatureAmount int) *Forest {
	rand.Seed(time.Now().UnixNano())
	forest := &Forest{}
	forest.Trees = make([]*Tree,treesAmount)
	prog_counter := uint32(0)
	scheduleoptimise := make(chan bool, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		scheduleoptimise <- true
	}
	var wg sync.WaitGroup 
	for i:=0;i<treesAmount;i++{
		<- scheduleoptimise
		wg.Add(1)
		go func(x int){
			fmt.Printf(">> Building tree %v...\n", x)
			forest.Trees[x] = BuildTree(inputs,labels,samplesAmount,selectedFeatureAmount)
			//fmt.Printf("<< %v the %vth tree is done.\n",time.Now(), x)
			prog_counter = atomic.AddUint32(&prog_counter, 1)
			fmt.Printf("Training progress %.0f%%\n",float64(prog_counter) / float64(treesAmount)*100) 
			wg.Done()
			scheduleoptimise <- true
		}(i)
	}

	wg.Wait()

	fmt.Println("Training done...")
	return forest
}

func DefaultForest(inputs [][]interface{},labels []string, treesAmount int) *Forest {
	m := int( math.Sqrt( float64( len(inputs[0]) ) ) ) 
	n := int( math.Sqrt( float64( len(inputs) ) )  )
	return BuildForest(inputs,labels, treesAmount,n,m)
}

// Predict the class of the input
func (self *Forest) Predict(input []interface{}) string{
	counter := make(map[string]float64)
	for i:=0;i<len(self.Trees);i++{
		tree_counter := PredictTree(self.Trees[i],input)
		total := 0.0
		for _,v := range tree_counter{
			total += float64(v)
		}
		for k,v := range tree_counter{
			counter[k] += float64(v) / total
		}
	}

	max_c := 0.0
	max_label := ""
	for k,v := range counter{
		if v>=max_c{
			max_c = v
			max_label = k
		}
	}
	return max_label
}

// Save a forest to disk
func (forest *Forest) Save(fileName string) {
	out_f, err:=os.OpenFile(fileName,os.O_CREATE | os.O_RDWR,0777)
	if err!=nil{
		panic("failed to create "+fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	encoder.Encode(forest)
}

// Load a forest from disk
func LoadForest(fileName string) *Forest {
	in_f ,err := os.Open(fileName)
	if err!=nil{
		panic("failed to open "+fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	forest := &Forest{}
	decoder.Decode(forest)
	return forest
}


