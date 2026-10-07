import {Component, Input, OnDestroy, OnInit} from '@angular/core';
import {
  circle,
  divIcon,
  icon,
  latLng,
  marker,
  polygon,
  tileLayer,
  Map,
  Icon,
  point,
  LeafletEvent,
  polyline, layerGroup, LayerGroup
} from "leaflet";
import {HttpClient} from "@angular/common/http";
// @ts-ignore
import * as Papa from "papaparse";
import {
  concatMap,
  from,
  groupBy,
  map,
  mergeMap,
  Observable,
  of,
  Subscription, tap,
  toArray,
  zip
} from "rxjs";
import {FlatInfo} from "../../types/flat-info.type";
import {Stop} from "../../types/stop.type";
import {ShapeCsvType, ShapeType} from "../../types/shape.type";
import {StopTime, TripsPerStop} from "../../types/stop-time.type";
import {Trip} from "../../types/trip.type";

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css']
})
export class MapComponent implements OnInit, OnDestroy {

  @Input() shoudLoadData: boolean = true;

  layersControl = {
    baseLayers: {},
    overlays: {}
  }

  options = {
    layers: [
      //tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18, attribution: '...' }),
      tileLayer('https://maps.wien.gv.at/basemap/bmaphidpi/normal/google3857/{z}/{y}/{x}.jpeg', {maxZoom: 19}),
    ],
    zoom: 13,
    center: latLng(48.210033, 16.363449)
  };

  tripsPerStop$!: Observable<{ [key: string]: string[] }>;
  shapesPerTrip$!: Observable<{ [key: string]: string }>;
  shapes$!: Observable<{ [key: string]: ShapeType[] }>;

  tripsPerStop: { [key: string]: string[] } = {};
  shapesPerTrip: { [key: string]: string } = {};
  shapes: { [key: string]: ShapeType[]; } = {};

  tripsPerStopSubscription!: Subscription;
  shapesPerTripSubscription!: Subscription;
  shapesSubscription!: Subscription;
  stopsSubscription: Subscription | undefined;
  flatSubscription: Subscription | undefined;


  pLayerGroup: LayerGroup = layerGroup();

  stopsLoaded = false;
  stopTimesLoaded = false;
  tripsLoaded = false;
  shapesLoaded = false;

  flatsLoaded = false;

  // Master array for all fetched flats
  allFlats: FlatInfo[] = [];
  flatLayerGroup: LayerGroup = layerGroup();
  
  // Filter flags
  filterGoodDealsOnly = false;
  filterWellConnectedOnly = false;


  constructor(private http: HttpClient) {
  }

  ngOnInit(): void {
    if (this.shoudLoadData) {
      const stopTimes$ = this.http
      .get('/2_construction/assets/data/wienerlinien/stop_times.txt', {responseType: 'text'})
        // .get('../../../2_construction/assets/data/wienerlinien/stop_times.txt', {responseType: 'text'})
        .pipe(
          map<string, StopTime[]>(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data)
        );

      this.tripsPerStop$ = stopTimes$
        .pipe(
          concatMap((arr) => from(arr)),
          map<StopTime, TripsPerStop>((e) => ({
            meta_stop_id: e.stop_id.substring(0, this.nthIndex(e.stop_id, ':', 3)),
            trip_id: e.trip_id,
          })),
          groupBy(data => data.meta_stop_id),
          mergeMap((group) => zip(of(group.key), group.pipe(map(el => el.trip_id), toArray()))),
          toArray(),
          map(data => data.reduce((obj: { [key: string]: string[] }, item) => {
            obj[item[0]] = item[1];
            return obj
          }, {})),
        );

      this.tripsPerStop = {};
      this.tripsPerStopSubscription = this.tripsPerStop$.subscribe((data) => {
        this.tripsPerStop = data;
        this.stopTimesLoaded = true;
      });

      this.shapesPerTrip = {};

      const trips$ = this.http
        .get('/2_construction/assets/data/wienerlinien/trips.txt', {responseType: 'text'})
        // .get('../../../2_construction/assets/data/wienerlinien/trips.txt', {responseType: 'text'})
        .pipe(
          map<string, Trip[]>(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data)
        );

      this.shapesPerTrip$ = trips$.pipe(
        map(data => data.reduce((obj: { [key: string]: string }, item) => {
          obj[item.trip_id] = item.shape_id;
          return obj
        }, {}))
      )

      this.shapesPerTripSubscription = this.shapesPerTrip$.subscribe((data) => {
        this.shapesPerTrip = data;
        this.tripsLoaded = true;
      })

      this.shapes = {};
      this.shapes$ = this.http
        .get('/2_construction/assets/data/wienerlinien/shapes.txt', {responseType: 'text'})
        // .get('../../../2_construction/assets/data/wienerlinien/shapes.txt', {responseType: 'text'})
        .pipe(
          map<string, ShapeCsvType[]>(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data),
          concatMap((arr) => from(arr)),
          map<ShapeCsvType, ShapeType>((data) => ({
            shape_id: data.shape_id,
            shape_pt_lat: parseFloat(data.shape_pt_lat),
            shape_pt_lon: parseFloat(data.shape_pt_lon),
            shape_pt_sequence: parseFloat(data.shape_pt_sequence)
          })),
          groupBy((data) => data.shape_id),
          mergeMap((group) => zip(of(group.key), group.pipe(toArray()))),
          toArray(),
          map(data => data.reduce((obj: { [key: string]: ShapeType[] }, item) => {
            obj[item[0]] = item[1].sort((a, b) => a.shape_pt_sequence - b.shape_pt_sequence);
            return obj
          }, {})),
        )

      this.shapesSubscription = this.shapes$.subscribe((data) => {
        this.shapes = data;
        this.shapesLoaded = true;
      })
    }
  }

  ngOnDestroy() {
    if (this.stopsSubscription != undefined) {
      this.stopsSubscription.unsubscribe();
    }
    if (this.flatSubscription != undefined) {
      this.flatSubscription.unsubscribe();
    }

    this.shapesSubscription.unsubscribe();
    this.tripsPerStopSubscription.unsubscribe();
    this.shapesPerTripSubscription.unsubscribe();
  }

  nthIndex(str: string, pat: string, n: number) {
    let L = str.length, i = -1;
    while (n-- && i++ < L) {
      i = str.indexOf(pat, i);
      if (i < 0) break;
    }
    return i;
  }


  onMapReady(mapElement: Map) {
    if (this.shoudLoadData) {
      this.pLayerGroup.addTo(mapElement);
      this.flatLayerGroup.addTo(mapElement);

      const iconType1 = (label: string) => divIcon({
        iconSize: point(8, 8),
        className: "hover:!z-[1000] icon-stationen group relative flext items-center justify-center",
        html: `<div class="hidden group-hover:block absolute  whitespace-nowrap mt-4 -left-1/2 -ml-[50%] font-bold [text-shadow:0_4px_8px_rgba(0,0,0,0.12)]">${label}</div>`
      });

      const iconType2 = (label: string) => divIcon({
        iconSize: point(16, 16),
        className: "hover:!z-[1000] icon-unterkunft group relative flext items-center justify-center",
        html: `<div class="hidden group-hover:block absolute  whitespace-nowrap mt-4 -left-1/2 -ml-[50%] font-bold [text-shadow:0_4px_8px_rgba(0,0,0,0.12)]">${label}</div>`
      });

      const stops$: Observable<Stop[]> = this.http
        .get('/2_construction/assets/data/wienerlinien/stops.txt', {responseType: 'text'})
        // .get('../../../2_construction/assets/data/wienerlinien/stops.txt', {responseType: 'text'})
        .pipe(
          map(data => Papa.parse(data, {
            header: true,
            skipEmptyLines: true
          }).data)
        );

      this.stopsSubscription = stops$.subscribe((data) => {
        data.forEach((entry) => {
          const location = latLng(parseFloat(entry.stop_lat), parseFloat(entry.stop_lon));
          marker(location, {
            title: entry.stop_id,
            icon: iconType1(entry.stop_name)
          }).addTo(mapElement).on('click', (e) => this.handleMarkerClick(e))
        })
        this.stopsLoaded = true;
      })

      const flats$ = this.http
        .get<FlatInfo[]>('map_data.json', {responseType: 'json'});

      this.flatSubscription = flats$.subscribe((data) => {
        this.allFlats = data;
        this.renderFlats(); // Initial render with all data
        this.flatsLoaded = true;
      });

      // // 1. Read flats JSON data (pointing to your generated map_data.json)
      // const flats$ = this.http
      //   .get<FlatInfo[]>('map_data.json', {responseType: 'json'}); // Adjust path if placed in assets/

      // const formatter = new Intl.NumberFormat('de-DE', {style: 'currency', currency: 'EUR'});

      // // 2. Custom Leaflet icon to display actual vs predicted price
      // const iconFlat = (actualStr: string, predStr: string, isGoodDeal: boolean) => divIcon({
      //   iconSize: point(20, 20),
      //   className: "hover:!z-[1000] icon-unterkunft group relative flex items-center justify-center",
      //   html: `
      //     <div class="${isGoodDeal ? 'bg-green-600' : 'bg-blue-600'} w-4 h-4 rounded-full border-2 border-white shadow-md"></div>
      //     <div class="hidden group-hover:block absolute whitespace-nowrap mt-6 -left-1/2 -ml-[50%] font-bold text-xs bg-white/90 p-1 rounded shadow-lg border border-gray-200 text-gray-800 z-[2000]">
      //       <div>Actual: ${actualStr}</div>
      //       <div class="text-indigo-600">Predicted: ${predStr}</div>
      //     </div>
      //   `
      // });

      // // Add layer group for flats to map
      // this.flatLayerGroup.addTo(mapElement);

      // this.flatSubscription = flats$.subscribe((data) => {
      //   data.forEach((entry) => {
      //     // Extract latitude and longitude safely
      //     const latVal = entry.LATITUDE ? parseFloat(entry.LATITUDE) : entry.lat;
      //     const lonVal = entry.LONGITUDE ? parseFloat(entry.LONGITUDE) : entry.lon;

      //     if (latVal && lonVal) {
      //       const location = latLng(latVal, lonVal);
            
      //       // Extract prices
      //       const actual = entry.actual_price ?? (entry.PRICE ? parseFloat(entry.PRICE) : 0);
      //       const predicted = entry.predicted_price ?? actual;
      //       const isGoodDeal = entry.is_good_deal ?? false;

      //       const actualFormatted = formatter.format(actual);
      //       const predictedFormatted = formatter.format(predicted);

      //       // Create marker with Leaflet Popup on click
      //       const flatMarker = marker(location, {
      //         icon: iconFlat(actualFormatted, predictedFormatted, isGoodDeal)
      //       });

      //       // Bind click popup with detailed price comparison
      //       flatMarker.bindPopup(`
      //         <div class="p-2 text-sm font-sans">
      //           <h3 class="font-bold text-base mb-1 text-gray-800">Flat Details</h3>
      //           <div class="flex justify-between gap-4 py-1 border-b border-gray-100">
      //             <span class="text-gray-600">Actual Price:</span>
      //             <span class="font-semibold text-gray-900">${actualFormatted}</span>
      //           </div>
      //           <div class="flex justify-between gap-4 py-1 border-b border-gray-100">
      //             <span class="text-gray-600">Predicted Price:</span>
      //             <span class="font-semibold text-indigo-600">${predictedFormatted}</span>
      //           </div>
      //           <div class="mt-2 text-xs font-semibold ${isGoodDeal ? 'text-green-600' : 'text-amber-600'}">
      //             ${isGoodDeal ? '✓ Underpriced (Good Deal)' : '• Fairly / Overpriced'}
      //           </div>
      //         </div>
      //       `);

      //       flatMarker.addTo(mapElement);
      //     }
      //   });
      //   this.flatsLoaded = true;
      // });

  //     const flats$ = this.http
  //       .get<FlatInfo[]>('/2_construction/assets/data/flat_info.json', {responseType: 'json'})
  //       // .get<FlatInfo[]>('../../../2_construction/assets/data/flat_info.json', {responseType: 'json'});

  //     const formatter = new Intl.NumberFormat('de-DE', {style: 'currency', currency: 'EUR'});

  //     this.flatSubscription = flats$.subscribe((data) => {
  //       data.forEach((entry) => {
  //         const location = latLng(parseFloat(entry.LATITUDE), parseFloat(entry.LONGITUDE));
  //         marker(location, {
  //           //title: entry.HEADING,
  //           icon: iconType2(formatter.format(parseFloat(entry.PRICE)))
  //         }).addTo(mapElement).on('click', (e) => {
  //           console.log(entry)
  //         })
  //       })
  //       this.flatsLoaded = true;
      // })
    }
  }


  handleMarkerClick(e: LeafletEvent) {
    this.pLayerGroup.clearLayers();


    const title = e.target.options.title;
    const shortTitle = title.substring(0, this.nthIndex(title, ':', 3));

    const tripNames = this.tripsPerStop[shortTitle];

    if (tripNames == null) {
      console.log("Trips not found, have you loaded only a subset of the data?")
      return;
    }


    const shapeNames = tripNames.map((tripName: string) => this.shapesPerTrip[tripName]).filter((thing, i: number, arr) => {
      const thingName = arr.find((t) => t == thing);
      return thingName != undefined && arr.indexOf(thingName) === i;
    });

    const lines = shapeNames.map((shapeName: string) => this.shapes[shapeName]);


    lines.forEach((line) => {
      const pointList = line.map((linePoints: { shape_pt_lat: number; shape_pt_lon: number; }) => latLng(linePoints.shape_pt_lat, linePoints.shape_pt_lon));
      const el = polyline(pointList, {color: 'red'});
      this.pLayerGroup.addLayer(el);
    });


  }

  renderFlats() {
    // Clear previous markers
    this.flatLayerGroup.clearLayers();

    const formatter = new Intl.NumberFormat('de-DE', {style: 'currency', currency: 'EUR'});

    // Define modern badge icon
    const iconFlat = (actualStr: string, predStr: string, isGoodDeal: boolean, sizeStr: string, roomsStr: string) => divIcon({
      iconSize: point(80, 30),
      iconAnchor: point(40, 15),
      className: "custom-flat-marker group relative",
      // html: `
      //   <div class="${isGoodDeal ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-slate-800 hover:bg-slate-900'} 
      //               text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg border-2 border-white 
      //               flex items-center justify-center transition-transform transform group-hover:scale-110 cursor-pointer">
      //     ${actualStr}
      //   </div>
        
      //   <div class="hidden group-hover:block absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 
      //               bg-white text-gray-800 p-2.5 rounded-lg shadow-xl border border-gray-200 z-[3000] pointer-events-none">
      //     <div class="font-bold text-xs text-gray-500 uppercase tracking-wide mb-1">Price Prediction</div>
      //     <div class="flex justify-between text-xs mb-1">
      //       <span>Actual:</span>
      //       <span class="font-bold">${actualStr}</span>
      //     </div>
      //     <div class="flex justify-between text-xs mb-1">
      //       <span>Predicted:</span>
      //       <span class="font-bold text-indigo-600">${predStr}</span>
      //     </div>
      //     <div class="text-[10px] font-semibold mt-1.5 pt-1 border-t border-gray-100 ${isGoodDeal ? 'text-emerald-600' : 'text-amber-600'}">
      //       ${isGoodDeal ? '✓ Underpriced (Good Deal)' : '• Fairly / Overpriced'}
      //     </div>
      //   </div>
      // `
      html: `
        <div class="${isGoodDeal ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-slate-800 hover:bg-slate-900'} 
                    text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg border-2 border-white 
                    flex items-center justify-center transition-transform transform group-hover:scale-110 cursor-pointer">
          ${actualStr}
        </div>
        
        <!-- Single Hover Preview Card -->
        <div class="hidden group-hover:block absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-52 
                    bg-white text-gray-800 p-2.5 rounded-lg shadow-xl border border-gray-200 z-[3000] pointer-events-none">
          <div class="font-bold text-xs text-gray-500 uppercase tracking-wide mb-1">Flat Details</div>
          <div class="flex justify-between text-xs mb-1">
            <span class="text-gray-500">Area:</span>
            <span class="font-bold">${sizeStr}</span>
          </div>
          <div class="flex justify-between text-xs mb-1">
            <span class="text-gray-500">Rooms:</span>
            <span class="font-bold">${roomsStr}</span>
          </div>
          <div class="flex justify-between text-xs mb-1 pt-1 border-t border-gray-100">
            <span>Actual:</span>
            <span class="font-bold">${actualStr}</span>
          </div>
          <div class="flex justify-between text-xs mb-1">
            <span>Predicted:</span>
            <span class="font-bold text-indigo-600">${predStr}</span>
          </div>
          <div class="text-[10px] font-semibold mt-1 pt-1 border-t border-gray-100 ${isGoodDeal ? 'text-emerald-600' : 'text-amber-600'}">
            ${isGoodDeal ? '✓ Underpriced (Good Deal)' : '• Fairly / Overpriced'}
          </div>
        </div>
      `
    });

    // Apply active filters
    const filteredFlats = this.allFlats.filter(flat => {
      if (this.filterGoodDealsOnly && !flat.is_good_deal) return false;
      if (this.filterWellConnectedOnly && !flat.is_well_connected) return false;
      return true;
    });

    // Render filtered markers
    filteredFlats.forEach(entry => {
      const latVal = entry.LATITUDE ? parseFloat(entry.LATITUDE) : entry.lat;
      const lonVal = entry.LONGITUDE ? parseFloat(entry.LONGITUDE) : entry.lon;

      if (latVal && lonVal) {
        const actual = entry.actual_price ?? (entry.PRICE ? parseFloat(entry.PRICE) : 0);
        const predicted = entry.predicted_price ?? actual;
        const isGoodDeal = entry.is_good_deal ?? false;

        const actualFormatted = formatter.format(actual);
        const predictedFormatted = formatter.format(predicted);

        const sizeStr = entry.estate_size ? `${entry.estate_size} m²` : 'N/A';
        const roomsStr = entry.number_of_rooms ? `${entry.number_of_rooms}` : 'N/A';

        const flatMarker = marker(latLng(latVal, lonVal), {
          icon: iconFlat(actualFormatted, predictedFormatted, isGoodDeal, sizeStr, roomsStr)
        });

        this.flatLayerGroup.addLayer(flatMarker);
      }
    });
  }

  applyFilters() {
    this.renderFlats();
  }
}
