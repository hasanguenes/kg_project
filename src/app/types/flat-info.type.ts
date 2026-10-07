export type FlatInfo = {
  LATITUDE: string
  LONGITUDE: string
  // PRICE: string
  lat?: number
  lon?: number
  PRICE?: string
  actual_price?: number
  predicted_price?: number
  is_good_deal?: boolean,
  is_well_connected?: boolean
  has_direct_access?: boolean
  estate_size?: number
  number_of_rooms?: number
}
