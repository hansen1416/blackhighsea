<template>
    <div>
        <div class="bb">
            <span>Hi</span>
        </div>
    </div>
</template>
<script lang="ts">
import { defineComponent } from "vue";
// import axios, {AxiosResponse} from "axios";

export default defineComponent({
    components: {},
    data() {
        return {
            stockList: [],
        };
    },
    computed: {},
    // created() {

    // },
    methods: {},
});
</script>
<style lang="scss">
$anime-time: 8s;

$box-size: 80px;
$clip-distance: 0.05;
$clip-size: $box-size * (1 + $clip-distance * 2);
$path-width: 2px;

$main-color: #69ca62;

%full-fill {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
}

.bb {
    @extend %full-fill;
    width: $box-size;
    height: $box-size;
    margin: auto;
    // background: $codepen-logo-path no-repeat 50% / 70% rgba(#000, .1);
    color: $main-color;
    box-shadow: inset 0 0 0 1px rgba($main-color, 0.5);
    text-align: center;
    line-height: $box-size;

    span {
        // vertical-align: middle;
    }

    &::before,
    &::after {
        @extend %full-fill;
        content: "";
        // z-index: -1;
        margin: -1 * $clip-distance * 100%;
        box-shadow: inset 0 0 0 $path-width;
        animation: clipMe $anime-time linear infinite;
    }

    &::before {
        animation-delay: $anime-time * -0.5;
    }

    // for debug
    &:hover {
        &::after,
        &::before {
            background-color: rgba(#f00, 0.3);
        }
    }
}

@keyframes clipMe {
    0%,
    100% {
        clip: rect(0px, $clip-size, $path-width, 0px);
    }
    25% {
        clip: rect(0px, $path-width, $clip-size, 0px);
    }
    50% {
        clip: rect($clip-size - $path-width, $clip-size, $clip-size, 0px);
    }
    75% {
        clip: rect(0px, $clip-size, $clip-size, $clip-size - $path-width);
    }
}

*,
*::before,
*::after {
    box-sizing: border-box;
}
</style>
