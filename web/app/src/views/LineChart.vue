<template>
  <section>
    <div class="box">
      <canvas ref="chart"></canvas>
    </div>
  </section>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import axios, {AxiosResponse} from "axios"
import Chart from 'chart.js'

interface Data{
  date: string[];
  real_high: number[];
  real_low: number[];
  pred_high: number[];
  pred_low: number[];
  next_high: number[];
  next_low: number[];
}

export default defineComponent({
  props: {
    
  },
  data() {
    return {

    }
  },
  computed: {
    ctx(): CanvasRenderingContext2D {
      return (this.$refs.chart as HTMLCanvasElement).getContext('2d') as CanvasRenderingContext2D;
    },
    stockCode(): string {
      return this.$route.params.stock_code as string;
    },
  },
  methods: {
    drawChart(data: Data, y_max: number, y_min: number) {
      new Chart(this.ctx, {
            
              type: 'line',
              data: {
                  labels: data.date,
                  datasets: [
                  {
                    data: data.real_high,
                    borderColor: "rgb(233, 60, 88)",
                    fill: false,
                    pointRadius: 0,
                  }, 
                  { 
                    data: data.real_low,
                    borderColor: "rgb(46, 149, 98)",
                    fill: false,
                    pointRadius: 0,
                  },
                  {
                    data: data.pred_high,
                    borderColor: "rgba(233, 60, 88, .6)",
                    fill: false,
                    pointRadius: 0,
                  }, 
                  { 
                    data: data.pred_low,
                    borderColor: "rgba(46, 149, 98, .6)",
                    fill: false,
                    pointRadius: 0,
                  },
                  {
                    data: data.next_high,
                    borderColor: "rgb(233, 60, 88)",
                    fill: false,
                    pointRadius: 5,
                    pointBackgroundColor: "rgb(233, 60, 88)",
                  },
                  {
                    data: data.next_low,
                    borderColor: "rgb(46, 149, 98)",
                    fill: false,
                    pointRadius: 5,
                    pointBackgroundColor: "rgb(46, 149, 98)",
                  },
                  ]
              },
              options: {
                  title: {
                      display: true,
                      text: '600104 Price prediction',
                  },
                  scales: {
                    yAxes: [{
                          display: true,
                          ticks: {
                              suggestedMin: y_min,
                              min: y_min,
                              max: y_max,
                          }
                    }]
                  },
                  legend: {
                    display: false,
                  }
                  // layout: {
                  //   padding: {
                  //     left: 0,
                  //     right: 0,
                  //     top: 0,
                  //     bottom: 0,
                  //   },
                  // },
              }
          });
    }
  },
  mounted() {

    axios.get(`${process.env.VUE_APP_PY_API}/prediction/${this.stockCode}`)
    .then((response: AxiosResponse) => {
      if(response.status == 200) {

        const data: Data = response.data;
        
        // pop empty data
        data.date.push('', '')
        
        data.real_high.pop()
        data.real_low.pop()

        const data_len = data.real_low.length;

        const limit = 50;

        data.date = data.date.slice(data_len - limit);
        data.real_high = data.real_high.slice(data_len - limit);
        data.real_low = data.real_low.slice(data_len - limit);
        data.pred_high = data.pred_high.slice(data_len - limit);
        data.pred_low = data.pred_low.slice(data_len - limit);

        data.real_high = data.real_high.map(x => x)

        // predicted prices, 
        // becasue predicted result has one more day, so limit will not out of range
        data.next_high = new Array(limit)
        data.next_high.push(data.pred_high[limit])
        data.next_low = new Array(limit)
        data.next_low.push(data.pred_low[limit])

        const high = Math.max(...[...data.real_high, ...data.pred_high])
        const low = Math.min(...[...data.real_low, ...data.pred_low])

        this.drawChart(data, 
        Math.ceil(high + (high - low)/5), 
        Math.floor(low - (high - low)/10),
        );
      }
    })
    .catch((e) => {
      console.info(e);
    });
  },


});
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
  section {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100vw;
    height: 100vh;

    .box {
      width: 1200px;
      height: 800px;
      margin-top: -50px;
      max-width: 80vw;
      max-height: 70vh;
    }

    color: rgb(233, 60, 88);
    color: rgb(46, 149, 98);
  }
</style>
